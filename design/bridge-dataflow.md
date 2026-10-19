# The bridge's data flow

Where each byte and word of the mesh bridge goes: who writes it, who waits on it, and what happens when it fails. The
reasons for the design are in [bridge-and-ipc.md](bridge-and-ipc.md) (the bridge), [recovery.md](recovery.md) (loss
and recovery) and [heterogeneity.md](heterogeneity.md) (link costs and shares); this page repeats none of them.

Conventions. The page describes the code at mesh `725a3dfb` (`MESH_VERSION` 117); the guard, MESH_VERSION 118; the
link (its lanes and the keepalive on them, the ports' opening, the stop), MESH_VERSION 124's
link layer (the structural rewrite's WP1, metal-microbench output_data/structural-20261018/DESIGN.md); programs on the
link's lanes, WP2's (§2, §3, §4, §5; the streams and relays deleted); the requests, stripes and ledger deleted at
MESH_VERSION 125 (metal-microbench output_data/structural-20261018/FINAL-BRIDGE.md). A file name with no directory is in
`rdma/`. A citation is `file function`. Line numbers are given only where the code is ambiguous. "Word" means an
8-byte atomic in the shared region. "Block" is `hdr.block * hdr.pgsz`. The bridge always starts with `block` = 1
(`mesh-flow.c main`: `mesh_layout(&geometry,pg,1,...)`), so a block is one 16 KiB page on Apple silicon, and a row is
one block.

## 1. The processes and memory

### Processes and threads

| process / thread | started by | does | waits on |
|---|---|---|---|
| `rdma/mesh-flow`, the bridge (one per node) | the launchd agent `io.mesh.bridge` (`bin/mesh-user.sh` writes it with `ProgramArguments` = mesh-flow and the canonical configuration plist, `KeepAlive`, `RunAtLoad`, stderr to `~/Library/Logs/io.mesh.bridge.log`); the keeper loads it at boot | everything below | — |
| the bridge's event loop (its main thread) | `mesh-flow.c main` | ports, control sockets, map, flow state machine, client slots, the record's available memory | `kevent64` with a 250 ms timeout; `EVFILT_USER` (`wake`), sockets, `NOTE_EXIT` on client pids, signals, kernel network events |
| `control_watch` thread | `main` | turns a change of `hdr.control` into a `wake()` of the event loop | `os_sync_wait_on_address(&M->control)` |
| port p's work queue `mesh.port` (serial dispatch queue) | `ports_found` | every verbs call except post and poll: open the device, register the window, make, connect and (at a re-pairing) destroy the link's lanes; prepare a flow's records and cells (no verbs call); let go of a flow's lanes | its jobs (`port_open`, `link_job`, `flow_job`, `flow_stop`) |
| port p's two progress threads `mesh.rdma.receive` (side 0) and `mesh.rdma.send` (side 1), QoS user-interactive | `port_threads`, at the port's first listed flow or its link's connect; never joined (they end with the process) | side 1 polls SEND cells' `ready` words, posts SENDs on their lanes and takes every lane's SEND completions; side 0 posts a listed flow's first receives, takes every lane's receive completions, stores landing words (only raising them) and reposts receives | spin while anything moves; parked on `hdr.bell[p]` (`port_progress`) |
| finder queue `mesh.ports` | `main` | reads the device list (`ports_find`) and the host name (`host_read`) | — |
| a client process (a rank, anything linked to libnccl-mesh or libmesh) | its user | attaches to the region, leases, binds, prepares, starts programs, publishes and consumes | its own landing words, `status[d].prepared` |
| libnccl-mesh session progress thread | `nccl-mesh.c session_open` | the silent-link watcher: follows landings, cancels a silent or cancelled session | `pthread_cond_timedwait`, 1 ms (`WATCH_NS`) between scans |
| the client's GPU work | the client's command buffers | publishes into SEND cells; spins on landing words | `mesh.metal mesh_wait` / `mesh_wait_bounded` |

### The shared region `/mesh0`

The bridge creates the region at start: `shm_unlink`, `shm_open(MESH_NAME, O_CREAT, 0666)`, `ftruncate`, `mmap`
(`mesh-flow.c main`). It lays the region out with `mesh.h mesh_layout`. Every section is page-aligned and the arena is
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

Every time a client or reader maps the region, it checks the region's `magic` and `version` (`MESH_VERSION`) and checks that `bridge_pid` is alive
(`mesh-dataflow.c mesh_open`). A region of another version is refused with `EINVAL`; a dead bridge gives `ESRCH`.

**Header words** (`mesh.h struct hdr`):

| word | written by | read / waited by | meaning |
|---|---|---|---|
| `magic`, `version`, `pgsz`, `block`, `rows`, `node`, offsets, `length`, `boot` | bridge at start (`main`); `magic` last | clients at attach | geometry; `node` is a 31-bit fold of `gethostuuid`; `boot` is the bridge's start time on the wall clock |
| `bridge_pid` | bridge (start; 0 at stop) | every client wait (`mesh_gone`) | liveness of the bridge |
| `serial` | `mesh_slot_take` (clients and the bridge's own leases) | — | slot generation counter |
| `control` | clients (`mesh-call.c mesh_transfers_start`, `mesh-dataflow.c mesh_retire`); bridge (on a rebuilt binary, at stop) | the bridge's `control_watch` | "look at the client slots now" |
| `map`, `map_off`, `map_nodes` | bridge (`map_put`, `map_grow`) | `mesh-dataflow.c mesh_records_read` (seqlock), `mesh_map_wait` | the map: an odd `map` value means a write is in progress |
| `health` | bridge (`health_put`) | `mesh_nodes_health` | a second seqlock for health-only record changes |
| `busy`, `idle_since` | bridge (`busy_tell`) | `mesh_busy`, `mesh-stat` | flows; wall-clock ns at which the count last reached 0 |
| `stopping` | nothing (WP1: a changed binary stops the bridge at once) | clients (`nccl-mesh.c ncclMeshBridgeStopping`; metal-microbench `mesh_rank.m`), deleted with WP3 and WP7 | — |
| `lanes[8][10]` | clients claim and release a lane's word (CAS 0 to the owner word of the slot whose program rides it; at the release, to 0, or to `MESH_LANE_DIRTY` alone where set); bridge (`lanes_release` sets DIRTY, `lanes_reap` lets go of a gone owner's, `link_make` clears DIRTY) | `lanes_leased` (`program_take`), `link_idle`, the clients' init (`mesh_lanes_free`) | a lane's lease (§3) |
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
| `status[d]`, `struct mesh_port_info` | bridge (`status_prepared`, `status_failed`, `program_paired`, `program_take`); the client zeroes it at take | `phase` (`MESH_PAIRING`, `MESH_PAIRED`, `MESH_STOPPED`), `code`, `domain`, `failed` (the node where the failure happened), `prepared` (set to the owner once the program is paired or failed: the client's start waits on it) |

**Inside a client's leases** (arena pages owned through rows). The bridge writes these as well, which goes beyond
the ownership sentence in bridge-and-ipc.md#region (see "Places where the code and bridge-and-ipc.md disagree"):

- **SEND cells** (`mesh.h struct mesh_send`, 256 B, 128-aligned). One column per slot of each SEND transfer: one
  cell for each active invocation, plus one terminal cell. All of them sit in one lease from `mesh_transfers_prepare`,
  followed by the program's `struct mesh_target`s. The client writes `request.wr_id = 1` (the producer argument)
  at prepare time and `ready` at each publication. The bridge writes `pair`, `span`, `request` (the chunk-0 work
  request), `successor`, `chunks`, `queue` (the lane it rides) and `port` (`mesh-program.c sends_configure`). Where a
  program sends the rows it receives (a relay rank), the bridge's receive thread also writes `ready` (§4).
- **Landing (completion) words and the cancellation record** (`mesh-collective.c mesh_host_inputs`). There is one
  `uint64_t` for each received chunk of each invocation, per destination, each destination starting on a 16-word
  boundary. The words of destination d form `cancel->ranges[d]` (`mesh.h struct mesh_cancellation`); every `tx[d].cancel`
  names that one record. `mesh_host_inputs` writes each received row's `mesh_delivery` (the word of invocation 0 and
  the per-invocation stride `8*count*chunks`). The bridge's progress threads store into the words (§4). The words are
  cancelled with `~0` (§5).
- **Rings and payload**: the client's own sections (`mesh-call.c mesh_section_create`). The NIC writes received
  payload into them by DMA. Senders write payload into them (`memcpy` on the host, Metal kernels on the GPU).
- **Transfer lists** (`struct mesh_transfer`, 48 B), leased per destination (`mesh-call.c mesh_lists_grow`).

**The window and the bridge's two mappings.** The arena is mapped twice in the bridge. `M` maps the whole region.
`wire.data` is an alias of the arena alone, at a 4 GiB-aligned base (`mesh-verbs.h wire_map`), because
`tbt_post_recv` loads only the low 32 bits of an SGE address. Every SGE address is in `wire.data` (`wire_span`). A client maps the region once, at its own address; its leases are region offsets. A GPU
sees the client's SEND-cell lease and its landing words as `MTLBuffer`s created with `newBufferWithBytesNoCopy` over
the client's mapping (`mesh-metal.m mesh_metal_transport_create`).

**The bridge's own arena leases**: the map's records (`map_grow`), every port's link pages (`link_reserve`:
`link_pages()`, one block for the keepalive's frames, for all
`MESH_PORTS`, taken at start).

### Verbs objects and who owns them

| object | per | made | destroyed | owner |
|---|---|---|---|---|
| context, PD | device (port) | `mesh-verbs.h device_up` on the port's queue (`port_open`) | never released by a call: `down_device` forgets the handles where the device failed with its link active; the kernel releases them at exit | `struct port.device` |
| MRs | device: the window in 1 GiB extents (`MESH_REGION`), `IBV_ACCESS_LOCAL_WRITE` only | `device_up`, before any QP exists; checked against `max_mr` (`EFBIG`) | the kernel, at exit | `device.regions[]` |
| the link's lanes: `MESH_LINK_LANES` (10) UC QPs on 2 CQs (receive `completion`, send `sent`), the CQs made once a context | port | `link_job(LINK_JOB_MAKE)` on hearing the peer's hello: `pair_up` (INIT); `LINK_JOB_CONNECT`: `queue_receive` (RTR), `queue_send` (RTS) for every lane | the next pairing's `link_make`: `pairs_down` (each QP; the CQs kept and drained), or `pair_forget` where the context is gone | `port.link.provider` |
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
 │       mesh_host_publish ->     │   │   signals, EVFILT_USER <- control_watch)       │       │             │
 │       cell.ready = 1+cycle     │   │ port p: work queue (verbs objects)             │       │             │
 │ GPU:  mesh_publish -> ready    │   │   send thread:  poll cell.ready -> post SEND ──┼─UC────┼─> RECV ─┐   │
 │       mesh_wait_bounded <- word│   │   recv thread:  poll CQ -> store word, repost  │ TB5   │ recv thread │
 │ mesh_transfers_wake -> bell[p] │   │   parked on hdr.bell[p]                        │       │ stores word │
 │ mesh_transfers_start -> control│   │ control socket TCP [fe80::%enN]:18519 ─────────┼─TCP───┼─ control    │
 └──────────┬─────────────────────┘   └──────────┬─────────────────────────────────────┘       └──────┬──────┘
            │ mmap                                │ mmap (M) + wire alias of the arena (4 GiB-aligned)  │
            ▼                                     ▼                                                     ▼
 ┌──────────────────────────────── /mesh0 on node A ───────────────────────────────┐     /mesh0 on node B
 │ hdr: magic version node boot bridge_pid control map health busy stopping bell[8] │
 │ planes | arena bits | page entries | buffers | link table[8] | client slots[64]  │
 │   slot: owner configured key depth cyclic ports digest tx[64] status[64]          │
 │ delivery[rows] (row -> landing word, stride)                                      │
 │ ARENA (window, MR-registered whole on every device):                              │
 │   client leases: rings, SEND cells+targets, landing words+cancel record, lists    │
 │   bridge leases: map records, lane pages                                          │
 └──────────────────────────────────────────────────────────────────────────────────┘
```

## 2. The control plane

### Ports and peers

1. **Ports are found.** `ports_find` lists `rdma_*` devices on the finder queue, and `ports_found` gives each a port
   index in the order found. This runs at start, and again when `network_heard` sees an unknown interface attach or
   its link come on. The port table entry is `link_set(port, 0, MESH_STOPPED, 0)`.
2. **The device opened** (`port_open` on the port's queue, guarded), at the start and at `KEV_DL_LINK_ON` /
   `KEV_DL_IF_ATTACHED` while the port has no peer, never on a clock: `device_up` (context, port active, PD, window
   MRs), then `device_gid`. A context that fails with the port active is forgotten with every object made on it (no
   verbs call) and the device opened afresh. In `port_opened`: a registration refusal, or any other failure, leaves the
   port down until its link next comes on; on success the port records `max_pairs`, LID, GID and bandwidth and sets the
   phase to `MESH_PAIRING`.
3. **Hello** (`port_hello`): an open port with no peer joins ff02::1 on the interface and sends a UDP `HELLO` (node,
   boot, block bytes, port index) to `[ff02::1%enN]:18519`, every `MESH_PAUSE_NS` (3 s), a paced emission that decides
   nothing; a port whose paired link was lost says it at once.
4. **Hello heard** (`hello_heard`). The higher node answers with its own HELLO. The lower node dials the higher's TCP
   listener at the address it heard from. Both ends `port_greet` (non-blocking, no keepalive: the peer's loss is the
   kernel's to tell) and send a `HELLO` over the socket. The first message on a control socket must be a `HELLO` with an
   equal block size (`port_heard`), otherwise `EPROTO`. Once greeted, the bridge sets `link_set(..., MESH_PAIRED)` and
   `pairings++`, sends every record in its map to the new peer, and makes its link's lanes (§1, `link_turn`).
5. **A connection to the listener from loopback** is no port's: it is closed.

### The map

- Each node has one record (`mesh.h struct mesh_record`): node, name, boot, sequence, window, memory, model, the port
  table (`mesh_link_info[8]`: peer, phase, code, bandwidth, pairings, pairs, max_pairs) and health (`booted`,
  `available`).
- `record_changed` builds this node's record with `sequence+1` and sends `RECORD` on every greeted port. It runs on
  any `link_set` that changes peer or phase, on a rename (`host_taken`), and from `health_turn`, which samples the
  available memory and runs every `MESH_HEALTH_NS` (10 s).
- `record_heard` keeps the newest record per node in `known[]`, apart from the map (RFC 2328 §13). It takes a record
  only if the record is newer by (boot, sequence), then forwards it on every other greeted port. A copy of a pruned
  node is put back only where a path reaches that node again; a node's later boot is always newer.
- Writing the record:
  - If the links (each port's peer, phase and bandwidth), name, boot, window, memory and model are unchanged
    (`record_links_same`), `health_put` writes it under the `health` seqlock, and the map serial does not move.
  - Otherwise `map_put` writes it under the `map` seqlock (odd while writing). `map_put` prunes nodes that no paired
    path reaches (BFS over `mesh_joined`), logs each pruned node, and writes only when the map
    differs.
- Routes: a member reads the records' whole map (one link per cable at the beta of the slowest port, `mesh_paired`)
  and takes `mesh-collective.c mesh_routes`'s path. Every member holding the same records computes the same route.
- Digests: the **program digest** (`client.digest`, set by `mesh_transfers_digest`) is the client's; the bridge reads
  none of it (the READY that carried it is deleted). libnccl-mesh's `map_digest` (FNV-1a over the planner version,
  wire, members, routes, link costs, trees and programs) travels in the session's position 0, whose agreement compares
  it (`session_agree`).

### Messages on a control socket

`struct message` carries `magic` (`XMAGIC` = `0x4d595048 + MESH_VERSION`, so bridges of different versions drop each
other: `control_heard` raises `EPROTO`), `kind`, `node`, `count`, `boot`, `code`, `block`, `port`, `bytes`, `lid`,
`gid` and `pair[10][2]`. A payload of `bytes` follows the message. Sends queue in the port's
buffer, and more than `MESH_UNSENT` (64 MiB) unsent loses the link with `ENOBUFS` (`port_say`). No message names a
program or a flow.

| kind | sender, when | receiver's effect (`port_heard` → `message_heard`) |
|---|---|---|
| HELLO | UDP: `port_hello` (every 3 s while unpaired), `hello_heard` (higher node's answer); TCP: `port_greet` | greets the port (peer, peer port index, LID, GID) |
| RECORD | `record_changed`, `record_heard` (forwarding), a newly greeted port (every held record) | `record_heard` |
| LINK | `link_turn` once the link's lanes are made (`count` 10: each lane's QP and PSN) | `link_heard`: every lane, once a pairing, before the connect (`LINK_JOB_CONNECT`); anything else `EPROTO` |
| FAREWELL | `main` at stop, with `code` = the signal, or -1 for a binary change | `port.parted = 1`, after which loss is reported as `ECONNRESET` "parted" |
| any other | — | `port_lost(EPROTO)` |

### Client slots: `clients_turn`

`clients_turn(liveness)` runs on every event-loop iteration. A liveness turn (`alive`) comes every 3 s.

1. A flow whose client owner is no longer owned, or whose slot's `configured` is not that owner, gets
   `flow_end(f)`: `flow_stop(code 0)`, nothing told.
2. For each slot: on a liveness turn, an owner whose pid is gone (`mesh_gone`) is retired (`mesh_retire`). A new
   owner gets a kqueue `NOTE_EXIT` watch; if the pid is already gone (`ESRCH`), it is retired. In the event loop,
   `NOTE_EXIT` retires every slot of that pid.
3. A slot with `configured == owner`, not yet in `accepted[s]`, and with no flow left from an earlier owner of the
   same slot, is accepted: `ports = 0`, and `program_take(client, owner, d)` runs for each destination with transfers.
4. Once no flow references a non-owned owner (`zombie == 0`), `mesh_retired_release` returns closed rows to the
   window, and `mesh_reclaim` frees their arena pages.

**Retiring** (`mesh-dataflow.c mesh_retire`, called by the client's `mesh_detach`, by `nccl-mesh.c session_retire` for
a revoke, or by the bridge):

1. CAS `owner` to `owner | MESH_RETIRING`.
2. Set `configured = 0` and every list `length` to 0.
3. On every row the client owns, set `closed = 1` if it heads a lease, and clear its `ROW_OWN` bit.
4. Set `owner = 0`: the slot is vacant at once.
5. Wake every `status[d].prepared` waiter, so a pending start returns `ECANCELED`.
6. `mesh_notify(control)`.

## 3. Flows

A flow is one program's traffic toward one far node on one port, for one stripe of the program's invocations
(`struct flow`): its client's slot (`owner`) and destination (`destination`), its queues (at most two, of one slot
each), queue k on lane `lane[k]` of the port's link, and the stripe j of n cables it carries. It holds no queue pair.

**Leases.** A program's lanes toward a node are a pair its client leased at its communicator's init, the same pair
index at both ends of each link (mesh-call.h `mesh_lanes_choose`, `mesh_lanes_claim`: WP3, WP4), named in its slot's
`tx[d].lane`. `program_take` (event loop) finds, among the cables paired with the node (`cables`, in `cable_rank`
order), those on which `lanes_leased` holds: the link up, each lane's lease word exactly the program's owner word (no
DIRTY), the lane served and held by no flow. The flows are stripe j of n on the n such cables. It fails the
destination as a value where the node is this one (EINVAL naming it), no cable reaches it (EHOSTUNREACH naming it), the
program is not a lane pair's shape (`tx.slots` other than 1, more than two queues, two queues on one lane: EINVAL), no
cable carries its lease (ENOLCK), or a port's table is full (E2BIG).

**Making.**

| step | where | what |
|---|---|---|
| `flow_make` | event loop (`program_take`) | the flow, its table index, its lanes' `holder` set to it (no other program can take them) |
| `flow_job` | port's queue | `receives_configure` (the records, no posting); `sends_configure` (the cells' chains, each cell's `pair` its lane's QP and `queue` the lane); `port_list` (in the port's table, `progressing`) |
| `flow_arm` | the receive thread's first look at the listed flow | each ring's first records posted (`ring_advance`): every record of a lane is posted by the thread that takes its completions |
| `flow_done` | event loop | a job's error fails the flow (domain 1); else `running`, and `program_paired` once every flow of the program and destination runs |

**Client status.** `program_take` sets `status[d].phase = MESH_PAIRING`. `program_paired` sets `MESH_PAIRED`, `code =
0`, then `prepared = owner` and wakes the waiter. `status_failed` sets `MESH_STOPPED`, `code`, `domain` and `failed`,
then `prepared = owner`. The client's start, `mesh_transfers_start`, waits on `prepared` (its windows are WP4's to
delete: a program is paired once its flows are listed, with no message, so the wait is short).

**The QP budget.** Per device `max_pairs = max_qp - 1` (10 on the TB5 devices); `port.committed` counts the link's 10
lanes, from their first making; no flow takes one. Four communicators ride a link at once (lanes 2 to 9 in pairs).

**`flow_free`** (event loop, once `flow_stop` reported `stopped == 1` and no job runs): unlinks the flow, `port.flows--`,
frees the table index (the next `flow_make` at that index bumps its generation), `flow_count--`, then `busy_tell`.

### Progress threads (`port_progress`, `port_pass`)

**Passes.**
- One pass of side 1 runs `flow_send` on every listed, progressing flow (its streams' SENDs on their lanes); then,
  while the link serves any lane (`link.served`), `link_send`: the link's SEND completions, each naming its lane by its
  QP's number (`link_lane`; a lane not served, or a QP of an earlier pairing, drops its own): a program's lane's frames
  retired through the lane's gate, lane 0's keepalive frame where the configuration states `keepalive`, a lane is leased
  and that long passed (bridge-and-ipc.md#keepalive).
- One pass of side 0 arms every listed flow not yet armed (`flow_arm`), then `link_receive` while the link serves any
  lane: a program's record landed (`record_landed`, to the lane's holder; a lane with none drops what comes), a
  keepalive frame's receive posted again as it lands.
- `port_pass`, `flow_send`, `flow_arm`, `record_landed`, `link_send` and `link_receive` are one
  translation unit (`mesh-link.c`), inlined into `port_pass` as in the measured build (its only calls are the verbs'
  function pointers and cold failure paths).

**Parking.**
- A thread with something to look for (`any`) that moves nothing for `MESH_IDLE_NS` (100 ms; counted in
  `MESH_IDLE_PASSES` = 4096-pass batches) parks on `bell[p]` with a 1 ms timeout.
- With nothing to look for, it parks at once, with a 3 s timeout.
- A bell ring wakes it immediately.

**Table changes.** Flows enter the port's table with `port_list` (release store, then a bell ring). They leave with
`port_unlist` and `port_settle`: `epoch++`, then wait until both threads report `seen >= epoch`. After that, no
thread reads the flow. A thread stores `seen` only when the epoch it read differs from the one it last told, so the
line of `high`, `epoch` and `seen` stays clean in both cores between table changes.

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
1. The client leases its rings (`ring_make`: `DEPTH * slot` bytes each way).
2. It binds the out ring as a SEND and the in ring as a receive toward b (`mesh_transfer_bind`, binding identity
   `1 + 2*rank + large`, `invocation_pages = slot / pgsz`, depth 16).
3. It prepares (`mesh_transfers_prepare(slots=1, invocations=32, depth=16)`): cells, targets, and each transfer's
   `first`.
4. It sets the digest, makes the program cyclic, and creates the Metal transport, which calls `mesh_host_inputs`
   (landing words, the cancellation record, the deliveries).
5. It starts the program (`mesh_transfers_start`); the bridge runs §3 with no message to b.
6. On both ends, `sends_configure` turns each SEND cell into a chain of chunk work requests over its invocation's ring
   slot. The chunk address is `transfer_row` = `row + ((t - begin) % depth) * invocation_pages / block + chunk`, and
   each chunk's SGE comes from `chunk_span`, in `wire.data`. Each stream is linked as t = 0..31, and the last cell's
   successor is the first, because the program is cyclic.
7. `receives_configure` builds the receive ring: one record per chunk per invocation. Each record's word comes from
   `receive_input`.
8. Position 0 of every channel is the members' agreement (`session_agree`). Data positions start at 1.

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
   (`cell->pair`, `cell->queue` the lane), one request per chunk. The cell's own `request.wr_id` stays 1, the producer's argument. When the chain is done,
   `stream->cell = successor`; on reaching the first cell again, `stream->cycle++`.
6. **SEND completion.** `link_send` polls the link's `sent` CQ, names the lane by the QP's number and sets its gate's
   `retired` from the `wr_id`. An error status makes the lane dirty (`lane_lost`) and raises `flow_lost(status, domain
   2)` on its holder. The device completes every SEND (`mesh-verbs.h struct mesh_queue`).
7. **Hardware credit.** TN3205's credit flow control holds each SEND until the peer has posted its matching receive
   (`sends_configure`). The bridge has no software credit protocol on a lane.

**Receiving (node b):**

8. **Receives posted ahead.** The receive thread's first look at the listed flow posts each ring's first records
   (`flow_arm`); then `ring_advance` posts record n once record `n - behind` has landed. `behind` spans the
   transfer's ring depth less one, divided by the stripes; for this session on one cable that is 15 invocations.
   Posting also
   stays within `capacity` = `min(receive_capacity, the lane's share of the completion CQ)` frames. Gating uses landings and
   capacity alone (lines 526-527). Nothing in the bridge waits for the local reader to consume a slot, or for a
   relay's onward SEND to complete, before posting into that slot again. Reuse is safe only under the clients'
   `LAG`/`DEPTH` credit (step 2).
9. **DMA.** The NIC writes the chunk into the posted SGE: b's ring slot `(at % 16)`, chunk k, inside b's window.
10. **Landing.** `link_receive` polls the link's `completion` CQ, names the lane by the QP's number and skips non-RECV
    opcodes; a completion of a lane with a holder goes to it (`record_landed`): `record = wr_id`, then
    `ring->landed++`, then it raises `*record->input` to `1 + record->cycle` (a compare-exchange that never lowers a
    word, so a cancellation's `~0` stands), and then calls `ring_advance`. `receive_input` chose the word:
    - **(a) a client row with a delivery inside this program's cancellation range for that destination**: the
      chunk's own word, at `device_input + device_stride * u + 8 * index * chunks` (index = slot where all slots share
      the rows);
    - **(b) no such word, at the record's last chunk, where the receiving program publishes that row** (a relay rank,
      DESIGN.md decision 1): the SEND cell `ready` of the row's publication, at `targets[index].cells + 256 * u`, with
      `release = 1`. Then `send_wake` rings the cell's port's bell if that send thread is parked. This is the index
      hand-off: the relay copies nothing, and its send thread posts the same rows onward (steps 4-7 on the relay).
    - **(c) otherwise**: the record's private `sink`.
11. **The client observes the landing.** A reader waits on the last chunk's word, because chunks of one receive land in
    order on one lane (`mesh_host_arrived`, `mesh_metal_receive_prepare`).
    - Host: `mesh_host_arrived(ctx, ch->in, at)`, an acquire load. Landed means `word != ~0 && word >= 1 + c`
      (`nccl-mesh.c landed`).
    - GPU: `mesh_remote_wait_bounded(inputs + ch->word + (at % 32) * ch->word_stride, c, polls)` spins with a
      system-scope fence until `word >= 1 + c` (`mesh.metal`), for at most `polls` loads; past them the position is
      written into violation words [2..4]. `~0` also satisfies the wait, and the caller tells the three apart.
12. **Consumption.** The host copies or combines from `ch->receiving + (at % 16) * slot`; `METAL_STAMP` mismatches
    are role violations. This frees the client to publish `at + LAG` (step 2).

## 5. Failure and cancellation data flow

### The `~0` landing word

`mesh.h mesh_cancel(m, cancel, d)` stores `UINT64_MAX` into every word of `cancel->ranges[d]` that is still 0. In a
cyclic program it stores into every word that is not already `~0`, because a reader of a later cycle would wait past a
landed `1 + c`. It also sets `cancel->requested = 1`. Every spin in `mesh.metal` and every host check treats `~0` as
"cancelled". Two parties cancel:
- **The bridge**: `flow_stop(code != 0)` while the owner is still owned, for the failed flow's destination only.
- **libnccl-mesh**: `cancel_all`, for every destination of the session. This happens on a landing that reads `~0`,
  on a GPU drain with nothing finishing for 10 s (`drain`), on a bounded
  wait's violation, and on a revoke. While a session stays failed, its progress thread cancels again every 10 ms
  (`RECANCEL_NS`).

### `flow_fail` and `flow_stop`

`flow_fail(f, code, domain, where)` (event loop): `ended = 1`; `flow_stop`; every other flow of the same owner and
destination (its other cables) failed alike. Nothing is told the peer: its program ends by what ends it at its own end.

`flow_stop`, on the port's queue:

1. `lanes_release`: the flow's lanes' `holder` cleared, then `port_unlist` (once neither progress thread reads it, no
   completion of its lanes is taken as its own, and its rings and its lanes' gates stand still); a lane whose ring has
   receives posted and not landed, or whose gate has SENDs posted and not completed, has positions outstanding and its
   lease word gets `MESH_LANE_DIRTY`.
2. If `code != 0` and the owner is still owned: `mesh_cancel` that destination's words, then `status_failed` (phase
   `MESH_STOPPED`, `code`, `domain`, `failed = where`, `prepared` woken).
3. The census line of each queue (its lane's gate and its ring, `dirty` where marked), its storage
   freed; `stopped = 1`. No verbs call.

Domains: 1 = a post, ring or job error (`flow_lost(...,1)`, `flow_done`); 2 = an error completion status; 3 = a
poll error; 4 = a program-level failure or a link's loss.

### How each failure reaches the clients

| event | bridge path | what the local client sees | what the peer side sees |
|---|---|---|---|
| a data error on a flow (post, poll or completion) | progress thread `flow_lost` (CAS `failed`, `progressing = 0`, `wake`; an error completion also `lane_lost`), then `flows_turn` → `flow_fail` | its words for that destination become `~0`; status `MESH_STOPPED`, code and domain, `failed` = this node; the lane dirty | nothing told: its program's waits end by its own end's events (its caller's revoke) |
| the peer's program ended (`mesh_detach`, exit, revoke) | nothing here (no END): the peer's flow stops at its end; a lane it left with positions outstanding is DIRTY there | this end's waits on that program's positions end by its caller's revoke | — |
| a program not a lane pair's shape, no lease on any cable, no cable, a table full | `program_take`: `status_failed` (EINVAL, ENOLCK, EHOSTUNREACH) or `program_fail(E2BIG)` | the code, at start | — |
| a link lost (socket EOF or error, send failure, `EPROTO`, `KEV_DL_LINK_OFF` / `IF_DETACHED` → `ENETDOWN`, the port down until its link comes on) | `port_lost`: close the socket; `link_lost` (no verbs call: `served = 0`, the lanes left for the next pairing, a job of this pairing dropped as it ends); `flow_fail(code` (`ECONNRESET` if parted), `4, tell 0, where = peer)` on every flow of the port; `link_set(PAIRING` or `STOPPED)` → RECORD → map serial; a paired port says hello at once | words `~0`; status naming the peer; libnccl-mesh fails `ncclRemoteError` ("the link to node N was lost") and cancels all its destinations; the next session reads the new map | the same, on the other end |
| a link lane fails (an error completion or post on any lane) | the progress thread's `lane_lost`: the lane served no more, `broken`; `link_turn` marks it dirty; its holder fails (above); once nothing rides the link (`link_idle`: no lane leased or held), `link_turn` closes the control socket (`port.cycling`, `port_lost(ECONNRESET)`), and both ends pair afresh on hearing each other, every lane made again | its holder's words `~0` | the link lost, then paired again |
| a lane left dirty by a program (`MESH_LANE_DIRTY` in its lease word) | leased to no new program (`lanes_leased`); the link cycled as above once nothing rides it | — | the same |
| a link job fails (making or connecting) | `link_turn`: the port down until its link next comes on (`open = 0`, `port_lost(code)`); no retry on a clock | as a link lost | the link lost |
| the peer's bridge stops | peer: FAREWELL, then close; here: `parted = 1`, then EOF → `port_lost(ECONNRESET)` | `ECONNRESET`, logged "parted" | — |
| this bridge stops (a signal, or its binary changed: at once) | `main`: per port FAREWELL queued and the socket closed, `closing = 1`; `programs_stop(ESHUTDOWN)`: every program's words `~0` and its status, stored from the main thread; `bridge_pid = 0`; `mesh_notify(control)`; return: `atexit` `shm_unlink`, the kernel releasing every verbs object and thread. No queue, thread or verbs call is waited on | every program's words `~0`, status `ESHUTDOWN` naming this node. Later waits see `bridge_pid` 0 (`ESRCH`). A lease of the old region fails the next libnccl-mesh lease or session (`ncclMeshMemLive`, `attaches_stale`) | FAREWELL → `ECONNRESET` |
| a client process dies | `NOTE_EXIT`, or the liveness `mesh_gone` → `mesh_retire` → next turn `flow_end`; its leases let go at the next `link_turn` (`lanes_reap`, DIRTY kept) | — | nothing told |

**A changed binary.** On a liveness turn where the binary's inode or mtime changed (`stat(argv[0])`), the bridge stops
at once, as at a signal (0fb8eb54's rule): its programs fail as values and launchd starts the new binary.
`bin/mesh-user.sh` renames a new binary into place, and copies it anew when only the configuration changed, so a
configuration change also fires this. `hdr.stopping` is no longer written; its readers are deleted with WP4 and WP7.

## 7. Configuration operands

**The bridge's operands.** The bridge has one operand: the canonical plist that `bin/mesh-user.sh` writes from the
consumer's JSON, TOML or YAML document through `bridge_configuration.py load`, to
`~/Library/Application Support/io.mesh.bridge/configuration.plist`. Nothing has a default. `configuration_read` reads
up to 64 KiB of it.

| key (document) | canonical | effect on the data flow |
|---|---|---|
| `purpose` | `purpose` | none (required, for people) |
| `window_gib` / `window_percent` (exactly one) | `window` (bytes; a percentage is resolved against the node's memory on the node that installs it) | the arena is `window / pgsz` pages = rows (`mesh_layout`); every lease, ring, cell, landing word, map record and lane page comes from it; a window outside [1 page, 2^32 pages] is refused at start; every device registers it whole, and a device with fewer than `ceil(window / 1 GiB)` MRs refuses the port (`EFBIG`, with a 60 s to 30 min backoff) |
| (the document's stem) | `name` | logged at start only |

Changing the plist takes effect only when the bridge restarts. `mesh-user.sh` re-copies the binary on a configuration
change, so the binary-change stop (§5) applies.

**Fixed in the source, not operands** (each changes the data flow):

| constant | value | where | what it changes |
|---|---|---|---|
| `MESH_PAUSE_NS` | 3 s | `mesh.h` | a port's hellos while unpaired, liveness turns, the 3 s timeout of a progress thread with nothing to look for |
| `MESH_IDLE_NS` / `MESH_IDLE_POLL_NS` | 100 ms / 1 ms | `mesh-flow.c` | spin time before parking; poll period while parked |
| `MESH_PAIRS` / `MESH_LINK_LANES` / `MESH_FLOWS` | 16 / 10 / 256 | `mesh-bridge.h`, `mesh.h` | max QPs a flow; a link's lanes; flows a port |
| `MESH_PORTS` / `MESH_CLIENTS` / `MESH_DESTINATIONS` / `MESH_QUEUES` / `MESH_ORDERS` | 8 / 64 / 64 / 4 / 4096 | `mesh.h` | region tables |
| `MESH_UNSENT` | 64 MiB | `mesh-flow.c` | unsent control bytes before a link is dropped |
| `QD`, `MESH_REGION`, `MESH_BANK` | 4095, 1 GiB, 4 GiB | `mesh-verbs.h` | QP depth cap, MR extent, wire alias alignment |

**Operands of the clients that shape the bridge's traffic:**
- `NCCL_BUFFSIZE` (environment, default `MESH_BUFFSIZE` 4 MiB; `nccl-mesh.c setting`) sets the large slot through
  `mesh_wire_of`.
- `ncclMeshStream.polls` bounds GPU waits.
- metal-microbench `configs/structure.json` `flows.pairs_port` and `flows.pairs_session` plan held sessions (§3).

## 8. Where to look

| question | function |
|---|---|
| Who creates the region, and what is in it? | `mesh-flow.c main`, `mesh.h mesh_layout` |
| How does a client attach or take a slot? | `mesh-dataflow.c mesh_attach`, `mesh_slot_take`, `mesh_program_attach` |
| Where do leases come from? | `mesh-call.c mesh_section_create`, `mesh-dataflow.c mesh_rows_alloc`, `mesh_arena_alloc`, `mesh_lease_return` |
| How is a transfer declared? | `mesh-call.c mesh_transfer_bind`, `mesh_lists_grow` |
| Where are SEND cells and targets laid out? | `mesh-call.c mesh_transfers_prepare` |
| Where are landing words and the cancel record laid out? | `mesh-collective.c mesh_host_inputs` |
| How does a start reach the bridge, and what does it wait for? | `mesh-call.c mesh_transfers_start`; `mesh-flow.c control_watch`, `clients_turn`, `program_take` |
| How is a peer found and paired? | `port_open`, `port_opened`, `port_hello`, `hello_heard`, `port_greet`, `port_heard`, `link_turn` |
| How is the map written and read? | `record_changed`, `record_heard`, `map_put`, `health_put`; `mesh-dataflow.c mesh_records_read` |
| What does each message do? | `port_heard`, `message_heard`, `link_heard` |
| Exclusive or lanes? | `flow_make`, `flow_mode` |
| What happens at each flow state? | `flow_step`, `flow_done`, `flow_work`, `flows_turn` |
| How does a program get its lanes and cables? | `program_take`, `lanes_leased`, `cables`; relays are member ranks (bridge-and-ipc.md#routes) |
| How does a publication become a SEND? | `mesh-collective.c mesh_host_publish`; `mesh.metal mesh_publish`; `mesh-flow.c flow_send`, `sends_configure` |
| When are receives posted? | `receives_configure`, `ring_advance` |
| Which word does a landing store? | `receive_input`; `record_landed` |
| How does a relay hand a landing on? | `receive_input` (`release`), `send_wake` |
| How does the client see a landing? | `mesh-collective.c mesh_host_arrived`; `mesh.metal mesh_remote_wait_bounded`; `nccl-mesh.c landed`, `run` |
| How do the link's lanes work? | `link_job`, `link_turn`, `link_heard`, `link_remake`, `link_send`, `link_receive`, `lane_lost` |
| When do progress threads sleep or wake? | `port_progress`, `port_pass`, `port_list`, `port_unlist`, `mesh-call.c mesh_transfers_wake` |
| What happens on failure? | `flow_lost`, `lane_failed`, `flow_fail`, `flow_stop`, `lanes_release`, `flow_free`, `port_lost`, `link_turn` (the idle cycle) |
| Cancellation? | `mesh.h mesh_cancel`; `nccl-mesh.c cancel_all`, `progress_run` |
| ESTALE? | `program_stale`, `program_end`, `digest_apart`, `program_runs` |
| A client exits? | `clients_turn`, `flow_end`, `mesh-dataflow.c mesh_retire`, `mesh_retired_release` |
| Upgrade or stop? | `main` (the liveness block and teardown), `nccl-mesh.c ncclMeshBridgeStopping` |
| The QP budget? | `link_turn`, `mesh-verbs.h pair_up`, `device_up` (`max_pairs`) |
| The configuration? | `bridge_configuration.py load`, `mesh-flow.c configuration_read`, `bin/mesh-user.sh` |

## The loopback fabric

The operator: "our tensor parallelism code should be fast both on days when the 5 computers are a ring and also on days
when the 5 computers are a star" (metal-microbench docs/principles.md section 4), and "i cannot and will not precommit to
a specific ordering or routing!" (2026-10-03). Before the Studios arrive, the only way to run the bridge's relays and
forwarding (#routes, A47), the lanes routed flows ride, and the library's trees on 3 to 8 nodes is N bridges on one
host. mesh 800fe6bc deleted the fabric that did that ("which stood for a per-call fabric"): `rdma/loopverbs.c`, the
host executor `rdma/nccl-mesh-host.c` and the `mesh-flow-loop` and `host` targets, whose last readings were the NCCL
matrix on an 8-node line 2,346 of 2,346 and line 8, shuffled ring 8, star 8 and torus 2x4 104 of 104 each, and which
found the 4-node ring+tree deadlock (c65445c) and the cycle-0 defect (c5a4674), which "nothing at two nodes can show".
They are restored on the resident bridge as separate build targets; the deployed `mesh-flow` is the same source without
`MESH_LOOP_FABRIC` and behaves as before.

**loopverbs** (`rdma/loopverbs.c`): the ibverbs calls the bridge makes, over a shared segment (its name the loop
configuration's `fabric`): a byte ring a queue pair, its number the ring's, held while the queue pair lives (each making
moves the ring's generation). A SEND is copied into its destination's ring and completes as it is delivered (one to a
queue pair gone or made again is dropped and completes, as UC drops it); the owner's polls drain its ring into its
receives in the order posted, a SEND waiting for its receive as TB5's credit flow control holds it, `IBV_WC_LOC_LEN_ERR`
for a SEND longer than its receive; a move to `IBV_QPS_ERR` flushes the posted receives and SENDs
(`IBV_WC_WR_FLUSH_ERR`), as `pair_flush` expects; 10 queue pairs a device (TB5's); a SEND at most half a ring, 4 MiB,
the port's `max_msg_sz`; one link-local GID a port. Deviations from 800fe6bc^: the segment's name from the bridge's
configuration (`loop_fabric`), not an environment variable; device names `rdma_loop0` to `rdma_loop63`; a queue pair's
ring taken by compare-exchange of its `used` word, so numbers are reused only once free (800fe6bc^ took the next of
1,024 modulo, clobbering a live ring past 1,024 makings); the dropped SEND; the flush; rings of 8 MiB (1 MiB).

**A loop bridge** (`make -C rdma mesh-flow-loop`, `rdma/.build/loop/mesh-flow-loop CONFIGURATION`): mesh-flow.c built
with `MESH_LOOP_FABRIC` and loopverbs in place of `-lrdma`. Its configuration (the canonical property list, written by
`mesh-loop up`, not by `bridge_configuration.py`) adds `loop`: `region` (its shared memory object, in place of
`/mesh0`), `node` (its number, in place of the host UUID's fold), `fabric`, and `ports`, each a loopverbs `device` and its cable's two loopback ports (`listen`, this
end's; `peer`, the other end's). `configuration_read` refuses one without them (`loop_read`). Its ports are those
devices alone, in that order (`ports_find`); a port's turn opens its device and the cable's lower port dials the
higher's listener, which accepts it as the port's control socket (`port_opened`, the event loop's `TAG_LOOP`), in
place of a link's hello and the lower node's dial; the HELLO, records, map, lanes and flows
are the deployed bridge's; its listener and hello socket take any port. Its name is its configuration's (`host_read` reads no host name).

**A loop client** (`make -C rdma host`): `rdma/.build/host/libmesh.dylib` (libmesh with `MESH_LOOP_FABRIC`:
`mesh_loop_bind(region)` names the bridge a process attaches to, before its first attach) and `rdma/.build/host/libnccl-mesh.dylib` (nccl-mesh.c with `MESH_HOST_EXECUTOR` and nccl-mesh-host.c in
place of nccl-mesh-metal.m and mesh-metal.m: every function of nccl-mesh-metal.h run on the CPU in the walk's order,
transcribed from the kernels, its waits polls of the words the bridge stores, every wait bounded (the program's polls,
and at most `MESH_PLATFORM_NS`), so the Metal path's walk runs with no GPU kernel; the Metal 4 form's own steps are
refused as the classic form refuses them). Deviation from 800fe6bc^: HEAD's Metal half (bounded waits, the violation words, combine3,
counted all-to-all, split-KV partials, retain and contents) transcribed; the arithmetic is nccl-mesh.c's own
(`nccl_mesh_combine`, `nccl_mesh_premultiply`, `nccl_mesh_truncdiv`, hidden, where `MESH_HOST_EXECUTOR` is defined).

**The driver** (`rdma/.build/host/mesh-loop`, rdma/mesh-loop.c; metal-microbench b9dbc43^ tools/nccl_demo.py `loop`'s
part that needs no Python):
- `mesh-loop up FAMILY N WINDOW_MIB DIRECTORY`: a line, ring, star, complete map or a torus of two rows, of N (2 to 8)
  nodes, every cable two loopverbs devices and two free loopback ports; each node's configuration written, its bridge
  started (posix_spawn, its log in DIRECTORY), and the call returns once every bridge's map holds N nodes and its own
  cables are paired (`mesh_nodes_health`, `mesh_observe`, within 60 s).
- `mesh-loop relay DIRECTORY BYTES`: every ordered pair, one after another, a program of one transfer of BYTES from
  the one to the other (the routed pairs through the relays between, on lanes), every word checked at the receiver.
- `mesh-loop nccl DIRECTORY COUNT host|walk`: N ranks, all-reduce, all-gather, reduce-scatter, broadcast and reduce
  from every root, and a send/receive around the ring, COUNT float32 each, every value exact, on the host path or the
  Metal path's walk on the host executor.
- `mesh-loop down DIRECTORY`: SIGTERM to every bridge, the fabric unlinked.

**Reproduction targets** (800fe6bc^, its readings with the GPU walk on the host executor): the NCCL matrix on an 8-node
line 2,346 of 2,346; line 8, shuffled ring 8, star 8 and torus 2x4 104 of 104 each; ring 5 with routes 46 of 46. The
matrix itself (every datatype, op and root, metal-microbench b9dbc43^ tools/nccl_demo.py) is the consumer's driver's to
restore on these targets; `mesh-loop nccl` is its float32 core. The integration check: `mesh-loop up line 8 512 D`,
then `relay D 4194304` (56 pairs, the farthest 7 hops), `nccl D 65536 host` and `nccl D 65536
walk`, then the same on star 8, ring 5 and torus 8; the resident bridge's relays have never run on more than two nodes.

## Routed readiness

Scatter-add and an expert layer's dispatch and combine route each update to a destination known only at run time
(metal-microbench docs/principles.md section 4, "Routed experts and engrams are one partition": "unrouted slices
contribute exact zeros, so they are neither computed nor, where placement allows, loaded"). mesh 45902d8e (2026-09-14,
"Delete sparse routing and segmented indexed-add machinery") removed the readiness half of that from the library with
its caller; it is restored here on the numerical executor (`mesh-call.h`), where every function fires as its
arrivals come (`mesh_call_progress`): the routing directory, so an owner waits on the producers routed to it alone,
and active segments, so a slot past a count produced at run time is omitted (Megablox's bounded grid; mesh e1acaf7e^
design/active-segments.md). The sort that writes the directory and the segmented sums that read it stay the caller's
functions (the operator's separation of 2026-09-14); the executor owns only when each may run.

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

## Places where the code and bridge-and-ipc.md disagree

1. **Who writes in a client's slot and rows.**
   - #region (lines 103-104) and `mesh.h` say the bridge writes only the statuses, the ports and the map.
   - The bridge also writes into the client's leased rows: every SEND cell's `pair`, `span`, `request`, `successor`,
     `chunks`, `queue` and `port` (`sends_configure`), and every landing word (`record_landed`).
2. **When RECORD is sent.**
   - The #protocol table's RECORD row (line 219) omits the periodic send.
   - `health_turn` sends this node's record on every port every 10 s with its available memory, and `record_retell`
     its held record every 3 s.

Outside bridge-and-ipc.md: heterogeneity.md §1 (line 28) says a link's beta is "8e9 over the widest paired port's
bandwidth". `mesh-dataflow.c mesh_paired` and `mesh-plan.h` use the slowest port.
