# The bridge's data flow

Where each byte and word of the mesh bridge goes: who writes it, who waits on it, and what happens when it fails. The
reasons for the design are in [bridge-and-ipc.md](bridge-and-ipc.md) (the bridge), [recovery.md](recovery.md) (loss
and recovery) and [heterogeneity.md](heterogeneity.md) (link costs and shares); this page repeats none of them.

Conventions. The page describes the code at mesh `725a3dfb` (`MESH_VERSION` 117). A file name with no directory is in
`rdma/`. A citation is `file function`. Line numbers are given only where the code is ambiguous. "Word" means an
8-byte atomic in the shared region. "Block" is `hdr.block * hdr.pgsz`. The bridge always starts with `block` = 1
(`mesh-flow.c main`: `mesh_layout(&geometry,pg,1,...)`), so a block is one 16 KiB page on Apple silicon, and a row is
one block.

## 1. The processes and memory

### Processes and threads

| process / thread | started by | does | waits on |
|---|---|---|---|
| `rdma/mesh-flow`, the bridge (one per node) | the launchd agent `io.mesh.bridge` (`bin/mesh-user.sh` writes it with `ProgramArguments` = mesh-flow and the canonical configuration plist, `KeepAlive`, `RunAtLoad`, stderr to `~/Library/Logs/io.mesh.bridge.log`); the keeper loads it at boot | everything below | — |
| the bridge's event loop (its main thread) | `mesh-flow.c main` | ports, control sockets, map, flow state machine, client slots, requests, health | `kevent64` with a 250 ms timeout; `EVFILT_USER` (`wake`), sockets, `NOTE_EXIT` on client pids, signals, kernel network events |
| `control_watch` thread | `main` | turns a change of `hdr.control` into a `wake()` of the event loop | `os_sync_wait_on_address(&M->control)` |
| port p's work queue `mesh.port` (serial dispatch queue) | `ports_found` | every verbs call except post and poll: open and retire the device, register the window, make, modify and destroy QPs and CQs, configure flow sides, write a flow's ledger | its jobs (`port_turn`, `flow_job`, `lanes_job`, `flow_stop`) |
| port p's two progress threads `mesh.rdma.receive` (side 0) and `mesh.rdma.send` (side 1), QoS user-interactive | `port_threads`, at the port's first listed flow or lanes connect; kept until `port_still` | side 1 polls SEND cells' `ready` words, posts SENDs and takes SEND completions; side 0 takes receive completions, stores landing words and reposts receives | spin while anything moves; parked on `hdr.bell[p]` (`port_progress`) |
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
| `busy`, `idle_since` | bridge (`busy_tell`) | `mesh_busy`, `mesh-stat` | flows plus awaiting programs; wall-clock ns at which the count last reached 0 |
| `stopping` | bridge (`main`, once its binary changed) | clients (`nccl-mesh.c ncclMeshBridgeStopping`; metal-microbench `mesh_rank.m`) | "release held flows between executions" (§5) |
| `bell[8]` | clients (`mesh_transfers_wake`); bridge (`port_bell`, `send_wake`) | port p's progress threads, when parked | a futex-like word; this is not a hardware doorbell |

**A client slot** (`mesh.h struct mesh_client`, 64 slots):

| field | written by | meaning |
|---|---|---|
| `owner` | the client takes the slot with a CAS (`mesh_slot_take`); `mesh_retire` clears it | `(generation << 38) \| (slot << 32) \| pid`; 0 means vacant |
| `configured` | the client sets it with a CAS 0→owner (`mesh_transfers_start`); `mesh_retire` sets 0 | the program is started |
| `key` | the client (`mesh_slot_take`: FNV-1a of the execution's name, `mesh_key`) | the execution; it is the same on every member |
| `depth`, `cyclic`, `digest`, `targets`, `target_count` | the client (`mesh_transfers_prepare`, `_cyclic`, `_digest`) | ring depth; whether the program is cyclic (M30); program digest (0 = none); the program's publications sorted by row |
| `ports` | bridge (cleared at acceptance in `clients_turn`; a bit set per flow's port in `flow_take`) | the bells `mesh_transfers_wake` rings |
| `tx[d]` (64), `struct mesh_tx` | the client (`mesh_transfer_bind`, `_prepare`, `mesh_host_inputs`); the bridge for its own relay leases (`mirror`) | the destination `node`; the lists region offset `orders` with `length[4 queues x 2 directions]` and `room`; `cells` (region offset of this destination's first SEND cell); `cancel` (region offset of the cancellation record); `slots`, `invocations`, `count`, `once` |
| `status[d]`, `struct mesh_port_info` | bridge (`status_prepared`, `status_failed`, `program_paired`, `flow_take`); the client zeroes it at take | `phase` (`MESH_PAIRING`, `MESH_PAIRED`, `MESH_STOPPED`), `code`, `domain`, `failed` (the node where the failure happened), `prepared` (set to the owner once the program is paired or failed: the client's start waits on it) |

**Inside a client's leases** (arena pages owned through rows). The bridge writes these as well, which goes beyond
the ownership sentence in bridge-and-ipc.md#region (see "Places where the code and bridge-and-ipc.md disagree"):

- **SEND cells** (`mesh.h struct mesh_send`, 256 B, 128-aligned). One column per slot of each SEND transfer: one
  cell for each active invocation, plus one terminal cell. All of them sit in one lease from `mesh_transfers_prepare`,
  followed by the program's `struct mesh_target`s. The client writes `request.wr_id = 1` (the producer argument)
  at prepare time and `ready` at each publication. The bridge writes `pair`, `span`, `request` (the chunk-0 work
  request), `successor`, `chunks`, `queue` and `port` (`mesh-flow.c sends_configure`). On a relay, the bridge's receive
  thread also writes `ready` (§4).
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
`tbt_post_recv` loads only the low 32 bits of an SGE address. Every SGE address and every lane header is in `wire.data`
(`wire_span`, `lane_send`). A client maps the region once, at its own address; its leases are region offsets. A GPU
sees the client's SEND-cell lease and its landing words as `MTLBuffer`s created with `newBufferWithBytesNoCopy` over
the client's mapping (`mesh-metal.m mesh_metal_transport_create`).

**The bridge's own arena leases**: the map's records (`map_grow`), every port's lane pages (`lanes_reserve`:
`lane_pages()` = 7 pages a port, for all `MESH_PORTS`, taken at start), and one relay lease per relayed direction
(`lease_take`, a client slot whose owner pid is the bridge's own; `mirror`).

### Verbs objects and who owns them

| object | per | made | destroyed | owner |
|---|---|---|---|---|
| context, PD | device (port) | `mesh-verbs.h device_up` on the port's queue (`port_turn`) | `down_device`, only when the port is retiring | `struct port.device` |
| MRs | device: the window in 1 GiB extents (`MESH_REGION`), `IBV_ACCESS_LOCAL_WRITE` only | `device_up`, before any QP exists; checked against `max_mr` (`EFBIG`) | `down_device` | `device.regions[]` |
| lanes: 2 UC QPs and 2 CQs (receive `completion`, send `sent`) | port | `lanes_job(LANES_MAKING)`: `pair_up`, then INIT; `LANES_CONNECTING`: `pair_receive` (RTR), `pair_send` (RTS) | `lanes_job(LANES_CLOSING)`: `down_pair`; `generation++` | `port.lanes.provider` |
| exclusive flow: `queues*slots` UC QPs (at most `MESH_PAIRS` = 16) and 2 CQs | flow | `flow_work(JOB_PAIR)`: `pair_up`; `JOB_RECEIVE`: RTR; `JOB_RUN`: RTS | `flow_release`: `down_pair` | `flow.provider` |

All QPs are `IBV_QPT_UC` (`mesh-verbs.h pair_up`, line 202), with MTU 4096, a GRH with hop limit 1 and the port's GID
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
 │       mesh_remote_wait <- word │   │   recv thread:  poll CQ -> store word, repost  │ TB5   │ recv thread │
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
 │   bridge leases: map records, lane pages, relay leases                            │
 └──────────────────────────────────────────────────────────────────────────────────┘
```

## 2. The control plane

### Ports and peers

1. **Ports are found.** `ports_find` lists `rdma_*` devices on the finder queue, and `ports_found` gives each a port
   index in the order found. This runs at start, and again when `network_heard` sees an unknown interface attach or
   its link come on. The port table entry is `link_set(port, 0, MESH_STOPPED, 0)`.
2. **Port turn**, every `MESH_PAUSE_NS` (3 s) while the port is unpaired (`port_turn` on the port's queue):
   `device_up` (context, port active, PD, window MRs), then `device_gid`. In `port_opened`: a registration refusal
   backs off 60 s, doubling up to 30 min; another failure with the device open retires the port; on success the
   port records `max_pairs`, LID, GID and bandwidth, sets the phase to `MESH_PAIRING`, joins ff02::1 on the interface
   and sends a UDP `HELLO` (node, boot, block bytes, port index) to `[ff02::1%enN]:18519`.
3. **Hello heard** (`hello_heard`). The higher node answers with its own HELLO. The lower node dials the higher's TCP
   listener at the address it heard from. Both ends `port_greet` (keepalive 1 s idle, 1 s interval, 2 probes;
   non-blocking) and send a `HELLO` over the socket. The first message on a control socket must be a `HELLO` with an
   equal block size (`port_heard`), otherwise `EPROTO`. Once greeted, the bridge sets `link_set(..., MESH_PAIRED)` and
   `pairings++`, and sends every record in its map to the new peer.
4. **A connection to the listener from loopback** is a client request (§2, Requests), not a peer.

### The map

- Each node has one record (`mesh.h struct mesh_record`): node, name, boot, sequence, window, memory, model, the port
  table (`mesh_link_info[8]`: peer, phase, code, bandwidth, pairings, pairs, max_pairs) and health.
- `record_changed` builds this node's record with `sequence+1` and sends `RECORD` on every greeted port. It runs on
  any `link_set` that changes peer or phase, on a rename (`host_taken`), and from `health_turn`, which runs every
  `MESH_HEALTH_NS` (10 s), or at once on an urgent change of power, charge, memory pressure or thermal state.
- `record_heard` keeps the newest record per node in `known[]`, apart from the map (RFC 2328 §13). It takes a record
  only if the record is newer by (boot, sequence), then forwards it on every other greeted port. A copy of a pruned
  node is put back only where a path reaches that node again; an entry unheard for 60 s whose node the map no longer
  holds compares with nothing.
- Writing the record:
  - If the links (each port's peer, phase and bandwidth), name, boot, window, memory and model are unchanged
    (`record_links_same`), `health_put` writes it under the `health` seqlock, and the map serial does not move.
  - Otherwise `map_put` writes it under the `map` seqlock (odd while writing). `map_put` prunes nodes that no paired
    path reaches (BFS over `mesh_joined`), logs each pruned node with its last health, and writes only when the map
    differs.
- Routes: `mesh-dataflow.c mesh_route` copies the records, builds the whole map (one link per cable at
  `MESH_ALPHA_US` and the beta of the slowest port, `mesh_paired`), and takes `mesh-collective.c mesh_routes`'s path.
  Every bridge and member holding the same records computes the same route.
- Digests (four different things):
  - The **program digest**: `client.digest`, set by `mesh_transfers_digest`. Every READY carries it, and the bridges
    compare it (§5, ESTALE).
  - libnccl-mesh's `map_digest`: FNV-1a over the planner version, wire, members, routes, link costs, trees and
    programs. It is passed in as the program digest, and the session's position-0 agreement compares it again
    (`session_agree`).
  - `ready_shape`: FNV-1a of a READY without its stripe and route, which identifies the relay lease that a
    direction's stripes share.
  - `pair_hash`: a murmur3 finalizer of (key, the lower end, the higher end), which picks a stripe's cable.

### Messages on a control socket

`struct message` carries `magic` (`XMAGIC` = `0x4d595048 + MESH_VERSION`, so bridges of different versions drop each
other: `control_heard` raises `EPROTO`), `kind`, `node`, `count`, `key`, `boot`, `code`, `block`, `port`, `from`,
`to`, `failed`, `bytes`, `lid`, `stripe`, `gid` and `pair[16][2]`. A payload of `bytes` follows the message. Sends
queue in the port's buffer, and more than `MESH_UNSENT` (64 MiB) unsent loses the link with `ENOBUFS` (`port_say`).
A flow is named by `(port, key, near, far, stripe)` (`flow_on`); a message names it as `from` = the sender's end and
`to` = the far end.

| kind | sender, when | receiver's effect (`port_heard` unless noted) |
|---|---|---|
| HELLO | UDP: `port_opened` (each turn while unpaired), `hello_heard` (higher node's answer); TCP: `port_greet` | greets the port (peer, peer port index, LID, GID) |
| RECORD | `record_changed`, `record_heard` (forwarding), a newly greeted port (every held record) | `record_heard` |
| READY (payload `struct ready`, route, transfer list) | `ready_say`: `program_take` (direct: both ends; routed: lower end, one per stripe), `program_await` and `ready_heard` (a higher end taking a held or awaited flow), a relay forwarding (`ready_heard`) | `ready_heard`: pairs an existing flow, holds a new flow, hands it to an awaiting program, refuses it (FAIL), or builds a relay (§3) |
| PAIRS | `flow_done` after `JOB_PAIR`: an exclusive flow's `qp_num` and PSN per QP; a lane rider's count 0 and `code = generation<<32 \| slot` | `HEARD_PAIRS`; stores the peer QPs or the peer's table index and generation; `flow_mode` ANDs the exclusive choice (`EPROTO` if the flow had already decided otherwise) |
| POSTED | `flow_tell`: once the flow's receives are posted (state `POSTING`); on a relay, also only after its partner heard the next hop's POSTED | `HEARD_POSTED`, which lets the flow run; on a relay, also tries `flow_tell(partner)` |
| END | `flow_end`: the flow's client is gone or retired (`clients_turn`) | `flow_fail(ECONNRESET, domain 4, tell 0)`; a relay partner passes END on |
| FAIL | `flow_fail` with a `tell`; `ready_refuse` (a READY refused, as a value) | `flow_fail(ESTALE` if the code is ESTALE, else `ECANCELED`, domain 4); with ESTALE on a non-relay flow, also `program_stale` |
| LANES | `lanes_turn` once lanes are made (count 2, the peer QPs, `code` = header room); `lanes_reset` (count 0, `code` = why) | `lanes_heard`: stores the peer QPs and room, then connects; count 0, or new QPs while joined, resets this end's lanes (`lanes_reset`, untold) |
| FAREWELL | `main` at stop, with `code` = the signal, or -1 for a binary change | `port.parted = 1`, after which loss is reported as `ECONNRESET` "parted" |
| CHANNEL_OPEN / CONFIRM / REFUSE / DATA / WINDOW / SHUT / CLOSE | requests (below) | `channel_heard` |

### Client slots: `clients_turn`

`clients_turn(liveness)` runs on every event-loop iteration. A liveness turn (`alive`) comes every 3 s.

1. A flow whose client owner is no longer owned, or whose slot's `configured` is not that owner, gets
   `flow_end(f)`: END to the peer, then `flow_stop(code 0)`. Relay flows are excluded: their owner pid is the
   bridge's own.
2. For each slot: on a liveness turn, an owner whose pid is gone (`mesh_gone`) is retired (`mesh_retire`). A new
   owner gets a kqueue `NOTE_EXIT` watch; if the pid is already gone (`ESRCH`), it is retired. In the event loop,
   `NOTE_EXIT` retires every slot of that pid.
3. A slot with `configured == owner`, not yet in `accepted[s]`, and with no flow left from an earlier owner of the
   same slot, is accepted: `ports = 0`. If a held flow of its key names a different digest, `program_stale` runs.
   Otherwise `program_take(client, owner, d)` runs for each destination with transfers.
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

### Requests (channels)

A client connects to `[::1]:18519` and sends a `struct mesh_request` (`mesh-dataflow.c bridge_ask`). The bridge
answers with an int32: 0 once the channel is open, otherwise an errno (`asked`). A channel's bytes ride the
control sockets hop by hop:

- **Opening**: `CHANNEL_OPEN` goes to the first cable of the route's next hop (`channel_hop`). At the destination,
  `service_open` admits the request against `available_now()` (`admitted`, `ENOMEM`) and opens `hold`, `exec`,
  `nodeinfo` or a registered Unix socket.
- **Flow control**: each hop grants a 128 KiB window (`MESH_WINDOW`) and returns credit only as bytes leave
  (`end_drain`). `CHANNEL_DATA` carries at most 32 KiB.
- **Failure**: a lost port ends its channels (`requests_lost`), and `CHANNEL_CLOSE` reaches the other ends. Channels
  never touch the window or RDMA.

## 3. Flows

A flow is one program's traffic toward one far node on one port, for one stripe of the program's invocations
(`struct flow`). It has two sides, `side[MESH_SEND]` and `side[MESH_RECEIVE]`, each a (slot owner, destination index,
relay lease). On a client's flow both sides are that client's program; on a relay, the receive side belongs to one
flow of the pair and the send side to the other.

### Kinds of flow

| kind | made by | queue pairs | rows |
|---|---|---|---|
| direct client flow (one per cable to the node, stripe j of n on cable j in `cable_rank` order) | `program_take` → `flow_make(..., exclusive=1)` (line 3101) | exclusive if both ends have room, otherwise the lanes | the client's |
| routed client flow (lower end; one per stripe of `route_stripes`, on `stripe_cable`) | `program_take` → `flow_make(..., 0)` (line 3116) | lanes | the client's |
| held flow (the route ends here, and no local program of the key has taken it yet; `hold = 1`) | `ready_heard`, hops == 1 → `flow_make(..., ready.exclusive)` | per the READY, and room | none until `flow_take` |
| relay pair `f` (toward the previous hop) and `g` (toward the next hop), partners | `ready_heard`, hops > 1 → two `flow_make(..., 0)` | lanes | a bridge lease (`mirror`), shared by a direction's stripes (`relay_lease`) |

**Exclusive vs lanes.**
- An exclusive flow holds one UC QP per ring (`queues * slots`) and its own two CQs. Its receives are posted ahead of
  the data. Program data (a client's operands) rides exclusive flows only (bridge-and-ipc.md#lanes; the operator's
  words quoted there).
- Every flow without QPs of its own shares its port's two lanes: ring q rides lane `(q / slots) mod 2`
  (`ring_lane`). A lane's receiver posts a crossing's records only when the header naming them lands, which costs a
  wire round trip per crossing.
- In the current code, routed client flows, relay flows, and direct flows that did not get QPs ride the lanes.
- `flow_make` grants exclusive only if `qps <= MESH_PAIRS` (16) and
  `committed + qps + (lanes committed ? 0 : 2) <= max_pairs`. The lanes' two QPs are counted before any flow's. The
  peer's READY or PAIRS can only turn exclusive off (`flow_mode`), and only until the flow has `decided` (at
  `JOB_PAIR`).

### Making a flow: state machine (`flow_step`, `flow_done`, `flow_work`)

| state | entered | job run on the port's queue | leaves when | deadline (`flows_turn`: `ETIMEDOUT`, naming `far`) |
|---|---|---|---|---|
| `READIED` | `flow_make` | — | READY or PAIRS heard → `JOB_PAIR` (`decided = 1`) | `created + MESH_PEER_NS` (600 s) |
| (`JOB_PAIR`) | | exclusive: `pair_up` (QPs in INIT); lanes: nothing | done → send PAIRS → `PAIRING` | `now + MESH_PAIR_NS` (30 s); a held flow keeps 600 s |
| `PAIRING` | | | PAIRS heard, `!hold`, lanes up (or exclusive) → `JOB_RECEIVE` | as above |
| (`JOB_RECEIVE`) | | exclusive: `pair_receive` (RTR); `receives_configure` (exclusive: the first records posted by `ring_advance`); lane rider: `port_list(receiving)` | done → `POSTING`, `flow_tell` → POSTED | `now + 30 s` if the READY was heard and the flow is not a relay; otherwise `created + 600 s + 30 s` |
| `POSTING` | | | POSTED heard, lanes up → `JOB_RUN` | as above |
| (`JOB_RUN`) | | `sends_configure`; exclusive: `pair_send` (RTS); `port_list(progressing)` | done → `RUNNING`; `program_paired` once every flow of the program and destination runs and no await is left | none |
| `RUNNING` | | `JOB_CHANGE` if a relay side's lists arrive late: `port_quiet`, configure the side, list again | `flow_fail` / `flow_end` | — |
| `FAILED` | `flow_fail`, `flow_end` | `flow_stop` (§5) | `flow_free` once `stopped == 1` | — |

**Client status.** `flow_take` / `program_take` set `status[d].phase = MESH_PAIRING`. `program_paired` sets
`MESH_PAIRED`, `code = 0`, then `prepared = owner` and wakes the waiter. `status_failed` sets `MESH_STOPPED`, `code`,
`domain` and `failed`, then `prepared = owner`.

The client's start, `mesh_transfers_start`, waits on `prepared` for each destination. Each wait lasts at most 3 s at a
time, and the whole start at most `MESH_PEER_NS + MESH_PAIR_NS + MESH_PAUSE_NS`. It returns:
- 0 once every destination is paired;
- the first stopped destination's `code`;
- `ECANCELED` if its own slot was retired;
- `ESRCH` if the bridge is gone;
- `ETIMEDOUT` past the deadline.

**Routed pairs** (`program_take`, `program_await`). The lower node routes on its own map and sends one READY per
stripe. The higher node awaits those READYs (a `struct await`, counted in `busy`). It takes held flows at once and the
rest as `ready_heard` delivers them; each take answers with a READY back along the relays. The await's deadline is
600 s where this end's map routes to the lower node; otherwise it is 30 s from when the lower node was last reachable.
If the lower node was never heard of, or has been unreachable longer than that, the program fails `EHOSTUNREACH` at
once.

**Relays** (`ready_heard`, `relay_direction`, `mirror`):
1. A READY whose route continues past this node makes `f` and `g`.
2. `mirror` leases rows per transfer at `mesh_relay_slot` bytes. It binds each transfer twice over the same section,
   as a receive from the sender and as a SEND toward the far end. It then runs `mesh_transfers_prepare` (and
   `_cyclic`), and `mesh_transfers_digest` with the READY's digest.
3. The bridge sends the READY on, with the rest of the route.
4. The far end's answering READY arrives on `g`. `g` mirrors the reverse direction and forwards the payload unchanged
   to `f`'s peer.

**The QP budget.**
- Per device: `max_pairs = max_qp - 1` (10 on the TB5 devices: bridge-and-ipc.md#lanes).
- `port.committed` counts the lanes' 2 plus each exclusive flow's `qps`. `pair_up` refuses with `ENOSPC` beyond
  `device.pairs + count > max_pairs`. `port_pairs` publishes `pairs` and `max_pairs` in the port table and in each
  record.
- A libnccl-mesh session binds 2 queues × 1 slot per peer (`nccl-mesh.c session_open`), so 2 exclusive QPs per cable.
- The consumer plans held sessions against two numbers of its own, `pairs_port` (8) and `pairs_session` (2)
  (metal-microbench `configs/structure.json` `flows`, used by `tools/mesh/flows.py budget` and `held_sets`). The
  bridge does not read them. For comparison: a TB5 device's 10 usable QPs less the lanes' 2 leaves 8, and a session
  takes 2 per cable.

**Held flows** (two meanings):
1. The bridge's `hold`: a flow made from a READY before any local program of its key took it. It is paired (PAIRS
   sent) but posts nothing. `flow_take` (from `program_take` or `program_await`) clears `hold` and sets the sides,
   the ports bit and the status; the flow then runs. If no program takes it within 600 s of its creation, it fails
   `ETIMEDOUT`, naming the sender.
2. A client keeping its sessions, and so their RUNNING flows, past an execution (metal-microbench
   docs/placement.md#retention-and-protention-the-next-execution-planned-ahead). The bridge sees ordinary RUNNING
   flows, counted in `busy`; `stopping` asks the client to release them (§5).

**`flow_free`** (event loop, once `flow_stop` reported `stopped == 1` and no job runs):
- unlinks the flow and its partner;
- `port.flows--`;
- frees the table index (the next `flow_make` at that index bumps its generation, so a lane header naming the old
  generation is discarded);
- `committed -= qps` if the flow was exclusive;
- drops both sides' leases (a relay lease is retired once its last user drops it: `lease_free` → `mesh_retire`, and
  `unreleased = 1`);
- `flow_count--`, then `busy_tell`.

### Progress threads (`port_progress`, `port_pass`)

**Passes.**
- One pass of side 1 runs `flow_send` on every listed, progressing exclusive flow, then `lane_send` for both lanes
  and `lane_retire`.
- One pass of side 0 runs `flow_receive` on every listed exclusive flow, then `lane_receive`.

**Parking.**
- A thread with something to look for (`any`) that moves nothing for `MESH_IDLE_NS` (100 ms; counted in
  `MESH_IDLE_PASSES` = 4096-pass batches) parks on `bell[p]` with a 1 ms timeout.
- With nothing to look for, it parks at once, with a 3 s timeout.
- A bell ring wakes it immediately.

**Table changes.** Flows enter the port's table with `port_list` (release store, then a bell ring). They leave with
`port_unlist` and `port_settle`: `epoch++`, then wait until both threads report `seen >= epoch`. After that, no
thread reads the flow.

## 4. The life of one transfer

This section follows one libnccl-mesh large-class position `at` from rank A (node a) to rank B (node b) over a direct,
exclusive flow. Each step names the word written, its writer and its reader. The constants:
- `LAG = MESH_LAG = 8`, `DEPTH = 16` (ring slots), `POSITIONS = 32` (invocations of the cyclic program;
  `nccl-mesh.c`).
- Position `at` is invocation `t = at % 32` in cycle `c = at / 32`, and uses ring slot `at % 16`.
- The large slot is `NCCL_BUFFSIZE / 8` rounded to a block (512 KiB, `mesh-collective.c mesh_wire_of`), which is 32
  chunks of 16 KiB and 128 frames of 4 KiB. The large class rides queue 0 and the small class queue 1
  (`ring_bind(..., !large, ...)`). One slot per queue means ring index = queue = QP index.

**Setup, once per session** (`nccl-mesh.c session_open`):
1. The client leases its rings (`ring_make`: `DEPTH * slot` bytes each way).
2. It binds the out ring as a SEND and the in ring as a receive toward b (`mesh_transfer_bind`, binding identity
   `1 + 2*rank + large`, `invocation_pages = slot / pgsz`, depth 16).
3. It prepares (`mesh_transfers_prepare(slots=1, invocations=32, depth=16)`): cells, targets, and each transfer's
   `first`.
4. It sets the digest, makes the program cyclic, and creates the Metal transport, which calls `mesh_host_inputs`
   (landing words, the cancellation record, the deliveries).
5. It starts the program (`mesh_transfers_start`); the bridge runs §3.
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
5. **Posting.** For each chunk: if the gate has room (frames in flight `posted - retired` plus this request's
   `ceil(len/4096)` frames, at most `gate.capacity` = `min(send_capacity, sent CQ cqe / qps)`), post a copy of the
   prepared request with `IBV_SEND_SIGNALED` and `wr_id = queue << 48 | posted frames`. Do this on QP `cell->queue`,
   one request per chunk. The cell's own `request.wr_id` stays 1, the producer's argument. When the chain is done,
   `stream->cell = successor`; on reaching the first cell again, `stream->cycle++`.
6. **SEND completion.** `send_retire` polls the shared `sent` CQ and sets `gate.retired` from the `wr_id`. An error
   status raises `flow_lost(status, domain 2)`. The device completes every SEND (`mesh-verbs.h struct mesh_queue`).
7. **Hardware credit.** TN3205's credit flow control holds each SEND until the peer has posted its matching receive
   (`sends_configure`). The bridge has no software credit protocol on an exclusive flow.

**Receiving (node b):**

8. **Receives posted ahead.** `ring_advance` posts record n once record `n - behind` has landed. `behind` spans the
   transfer's ring depth less one, divided by the stripes; for this session on one cable that is 15 invocations.
   Posting also
   stays within `capacity` = `min(receive_capacity, completion CQ cqe / qps)` frames. Gating uses landings and
   capacity alone (lines 526-527). Nothing in the bridge waits for the local reader to consume a slot, or for a
   relay's onward SEND to complete, before posting into that slot again. Reuse is safe only under the clients'
   `LAG`/`DEPTH` credit (step 2).
9. **DMA.** The NIC writes the chunk into the posted SGE: b's ring slot `(at % 16)`, chunk k, inside b's window.
10. **Landing.** `flow_receive` polls the shared `completion` CQ and skips non-RECV opcodes. For each completion,
    `record = wr_id`, then `ring->landed++`, then it stores `*record->input = 1 + record->cycle` (release), and then
    calls `ring_advance`. `receive_input` chose the word:
    - **(a) a client row with a delivery inside this program's cancellation range for that destination**: the
      chunk's own word, at `device_input + device_stride * u + 8 * index * chunks` (index = slot where all slots share
      the rows);
    - **(b) no such word (a relay lease has no landing words or cancellation range), at the record's last chunk,
      where the receiving program publishes that row**: the SEND cell `ready` of the row's publication, at
      `targets[index].cells + 256 * u`, with `release = 1`. Then `send_wake` rings the cell's port's bell
      if that send thread is parked. This is the index hand-off: the relay copies nothing, and its send thread posts
      the same rows onward (steps 4-7 on the relay).
    - **(c) otherwise**: the record's private `sink`.
11. **The client observes the landing.** A reader waits on the last chunk's word, because chunks of one receive land in
    order on one QP (`mesh_host_arrived`, `mesh_metal_receive_prepare`).
    - Host: `mesh_host_arrived(ctx, ch->in, at)`, an acquire load. Landed means `word != ~0 && word >= 1 + c`
      (`nccl-mesh.c landed`).
    - GPU: `mesh_remote_wait(inputs + ch->word + (at % 32) * ch->word_stride, c)` spins with a system-scope fence
      until `word >= 1 + c` (`mesh.metal`). `~0` also satisfies the wait, and the caller tells the two apart. In the
      bounded form, `polls` loads are allowed, and past them the position is written into violation words [2..4].
12. **Consumption.** The host copies or combines from `ch->receiving + (at % 16) * slot`; `METAL_STAMP` mismatches
    are role violations. This frees the client to publish `at + LAG` (step 2).

**The same transfer on a lane** (routed client flows, relays, direct flows without room):

- **Sender** (`lane_send`, side 1). The send thread takes released cells of the lane's flows in table order (cursor)
  and builds a header in send-header frame `h = sent % 8`. The frame is at
  `lanes.page * pgsz + (lane * 10 + h) * 4096`, and is reused only once its SEND retired. Each entry is
  `{peer_slot, peer_generation, ring, chunks, bytes}`, bounded by the gate and the peer's room. The thread posts the
  header SEND, then every entry's chunk SENDs in order, with `wr_id = LANE_SENT | lane << 48 | frames`.
- **Receiver** (`lane_receive`, side 0). A completion with `wr_id = LANE_HEARD + lane` is a header, handled by
  `lane_heard`:
  - It checks the magic, the `sequence == received` and the room.
  - Each entry whose flow is the one the index and generation name, is still `receiving`, and rides this lane gets
    its ring's next `chunks` records posted. Any other entry gets as many discards into the lane's discard block.
  - It then posts the next header receive into receive header `8 + (received & 1)`.
  - A record's landing stores its word exactly as in step 10, and `lane_frames` drops.
- Lane senders and receivers keep no per-cell completion attribution, so the ledger keeps none for lane riders (§6).

## 5. Failure and cancellation data flow

### The `~0` landing word

`mesh.h mesh_cancel(m, cancel, d)` stores `UINT64_MAX` into every word of `cancel->ranges[d]` that is still 0. In a
cyclic program it stores into every word that is not already `~0`, because a reader of a later cycle would wait past a
landed `1 + c`. It also sets `cancel->requested = 1`. Every spin in `mesh.metal` and every host check treats `~0` as
"cancelled". Two parties cancel:
- **The bridge**: `flow_stop(code != 0)` while the owner is still owned, for the failed flow's destination only.
- **libnccl-mesh**: `cancel_all`, for every destination of the session. This happens on a landing that reads `~0`,
  on a destination silent for `MESH_SILENT_NS` (10 s) with positions pending (`progress_run`, the only user of
  `MESH_SILENT_NS`; the bridge never reads it), on a GPU drain with nothing finishing for 10 s (`drain`), on a bounded
  wait's violation, and on a revoke. While a session stays failed, its progress thread cancels again every 10 ms
  (`RECANCEL_NS`).

**Ambiguity: a late lane landing overwrites `~0`.** `lane_receive` (lines 1214-1218) stores a landed record's word
through `record->link`, whether or not that flow is still in the port's table. `flow_stop` unlists the flow and then
cancels its words (lines 1427-1430). A lane rider's records that were already posted can therefore land after the
cancel and overwrite `~0` with `1 + cycle`. `flow_release` waits for those frames (`lane_frames`) before it frees
anything. An exclusive flow cannot do this, because `flow_receive` runs only for listed flows. The library's
10 ms re-cancel (design/recovery.md Q16) is what restores `~0`. The bridge itself does not cancel again.

### `flow_fail` and `flow_stop`

`flow_fail(f, code, domain, tell, where, origin)` (event loop):

1. `state = FAILED`.
2. If `tell` is set, send END or FAIL (`code`, `failed = origin`) to the flow's peer.
3. `flow_stop`.
4. Fail the relay partner, telling END if the code is `ECONNRESET`, otherwise FAIL.
5. Fail every other flow of the same owner and destination (its other cables and stripes, and a relay lease's other
   stripes) with FAIL.
6. Drop the program's await.

`flow_stop`, on the port's queue:

1. `port_unlist`: the progress threads stop reading the flow.
2. If `code != 0` and the owner is still owned: `mesh_cancel` that destination's words, then `status_failed` (phase
   `MESH_STOPPED`, `code`, `domain`, `failed = where`, `prepared` woken).
3. `flow_release`: print the census, wait for the lane frames, `down_pair`, `ledger_write`, free the rings and
   streams.
4. Set `stopped` to 1, or to 2 to retry 3 s later.

Domains: 1 = a post, ring or job error (`flow_lost(...,1)`, `flow_done`, lanes); 2 = an error completion status; 3 = a
poll error; 4 = a protocol, timeout or program-level failure.

### How each failure reaches the clients

| event | bridge path | what the local client sees | what the peer side sees |
|---|---|---|---|
| a data error on a flow (post, poll or completion) | progress thread `flow_lost` (CAS `failed`, `progressing = 0`, `wake`), then `flows_turn` → `flow_fail(..., FAIL)` | its words for that destination become `~0`; status `MESH_STOPPED`, code and domain, `failed` = this node | FAIL → `ECANCELED`; its words `~0`; status naming this node |
| a window passes (peer never started, QPs never paired, held flow never taken) | `flows_turn` → `flow_fail(ETIMEDOUT, 4, FAIL, far, far)` | start returns `ETIMEDOUT` naming `far` | FAIL → `ECANCELED` |
| the peer's program ended (`mesh_detach`, exit, revoke) | peer: `clients_turn` → `flow_end`: END; here: `port_heard` → `flow_fail(ECONNRESET, 4, 0, origin)` | words `~0`, status `ECONNRESET` naming the peer; a relay forwards END | — |
| digests differ (ESTALE) | `ready_heard` / `clients_turn`: `ready_refuse(ESTALE)` or `flow_fail(ESTALE)`, then `program_stale` → `program_end` for every destination with transfers | every destination `ESTALE`, naming the node whose program differs; start fails (`start_failure`) | FAIL with ESTALE → `program_stale` there too (not for relays) |
| maps disagree on a cable or route; READY shape mismatch | `ready_heard` / `program_take` / `program_await`: `ready_refuse(EPROTO)` + `program_fail(EPROTO)` | `EPROTO` naming the refusing node | FAIL → `ECANCELED` |
| a table or device full | `flow_make` NULL → `program_fail(E2BIG)` / `ready_refuse(E2BIG)`; `pair_up` `ENOSPC` → `flow_done` → `flow_fail(ENOSPC, 1)` | `E2BIG` / `ENOSPC` | FAIL |
| a link lost (socket EOF or error, keepalive, send failure, `EPROTO`, pairing timeout, `KEV_DL_LINK_OFF` / `IF_DETACHED` → `ENETDOWN` with retire) | `port_lost`: close the socket; `lanes.up = 0`; `flow_fail(code` (`ECONNRESET` if parted), `4, tell 0, where = peer)` on every flow of the port; `requests_lost`; lanes CLOSING; `link_set(PAIRING` or `STOPPED)` → RECORD → map serial; on retire, `down_device` at a later `port_turn` once no flow or lane holds QPs | words `~0`; status naming the peer; libnccl-mesh fails `ncclRemoteError` ("the link to node N was lost") and cancels all its destinations; the next session reads the new map | the same, on the other end |
| lanes fail (completion error, header out of step, make or connect refused) | `lanes_lost` → `lanes_turn` → `lanes_reset`: LANES of count 0 to the peer; `flow_fail(code, 1)` for every lane rider at `POSTING` or later or in a receive/run/change job; lanes rebuilt 3 s later; the link and its exclusive flows go on | lane riders' words `~0` | LANES of 0 → its lanes reset, and its riders fail |
| the peer's bridge stops | peer: FAREWELL, then close; here: `parted = 1`, then EOF → `port_lost(ECONNRESET)` | `ECONNRESET`, logged "parted" | — |
| this bridge stops (signal, or a rebuilt binary with `busy == 0`) | `main`: `requests_stop`; per port FAREWELL, a 200 ms flush, `parted = 1`, `port_lost(ESHUTDOWN, retire)`; on each port's queue `flow_release`, then `port_still`, `down_pair(lanes)`, `down_device`; `bridge_pid = 0`; `mesh_notify(control)`; `atexit` `shm_unlink` | flows that remain fail with status `ECONNRESET`, `failed` = the peer node (lines 3429-3433 set `parted` before `port_lost`, so line 2805 reports `ECONNRESET` rather than `ESHUTDOWN`). Later waits see `bridge_pid` 0 (`ESRCH`). A lease of the old region fails the next libnccl-mesh lease or session (`ncclMeshMemLive`, `attaches_stale`) | FAREWELL → `ECONNRESET` |
| a client process dies | `NOTE_EXIT`, or the liveness `mesh_gone` → `mesh_retire` → next turn `flow_end` (END) | — | `ECONNRESET` |

**The `stopping` word.**
1. On a liveness turn where the binary's inode or mtime changed (`stat(argv[0])`), `main` sets `hdr.stopping = 1`
   and calls `mesh_notify(control)`. `bin/mesh-user.sh` renames a new binary into place, and copies it anew when only
   the configuration changed, so a configuration change also fires this.
2. A client holding sessions between executions reads it through `ncclMeshBridgeStopping` (metal-microbench
   `mesh_rank.m` `release_held`), destroys its communicators, and holds none afterwards.
3. Once `flow_count == 0` and `await_count == 0`, the bridge stops (`stop_reason = MESH_STOP_BINARY`) and launchd
   starts the new binary.
4. A signal stop never sets `stopping`.

## 6. The ledger

**Switch.** The configuration's `ledger = true` sets `ledger_kept` (`configuration_read`). The start log line then
says "ledgers kept".

**What is recorded, per flow.** `ledger_open` runs on the port's queue at the flow's first non-pair job, before
either side is configured.

| event | recorded by | fields (each time is `clock_ns` = `CLOCK_MONOTONIC` ns of that node) |
|---|---|---|
| SEND cell first seen released | `flow_send` / `lane_send` → `ledger_seen` | the cell, the stream cycle, `seen` |
| its SENDs posted | `ledger_stamp` | `first` (the first post), `posted` and `mark` (the gate's frames after the last post; an exclusive flow only) |
| its last SEND completed | `send_retire` → `ledger_retired` | `completed`, taken when the queue's retired frames reach `mark` (completions are FIFO per QP); a lane rider has none (0) |
| a record landed | `flow_receive` / `lane_receive` → `ledger_landed` | the record index, cycle, `landed` (after its word was stored) |

**Names.** Cells and records are named at configure time (`sends_configure`, `receives_configure`) by queue (ring),
invocation, position among that invocation's transfers on the queue, binding, chunks and chunk. Both ends' lists are
in the same order, so a cell and its records share (queue, invocation, position, cycle) across the link.

**Limits.** At most `MESH_LEDGER_EVENTS` (65,536) of each side, covering one pass, or every pass of a cyclic program.
Past the limit the ledger keeps the first events.

**Where it is written.** `ledger_write` runs inside `flow_release`, on the port's queue, after neither thread reads
the flow:
- The header line, `{"ledger":id,"key":...,"node","peer","port","near","far","stripe","stripes","exclusive","queues",
  "qps","invocations","cyclic","started","sends","send_room","landings","landing_room","file"}`, goes to the bridge's
  stderr log, `~/Library/Logs/io.mesh.bridge.log`.
- With more than `MESH_LEDGER_LINES` (2048) events, and stderr a regular file, the events go to
  `ledger-<key>-<node>-<peer>-<stripe>-<id>-<UTC>.jsonl` beside the log, which also starts with the header.
  Otherwise they go to the log.
- Event lines: `{"s":[id,queue,invocation,position,binding,cycle,chunks,seen,first,posted,completed]}` and
  `{"r":[id,ring,invocation,position,binding,chunk,chunks,cycle,landed]}`.

**Reader, `ledger.py`:**
- `read`: parses JSON from the log lines (dropping the `say` prefix), follows a header's `file` beside the log or in
  any directory given, and attaches events to the latest header of their ledger id.
- `peer_of`: the same key and stripe, with node/peer and near/far swapped; ties are broken by shared names.
- `crossings`: a cell counts only when all its chunks landed, at the last chunk's time; `--skip` drops early
  invocations.
- Clock offset (each node's own `CLOCK_MONOTONIC`, with an unknown offset): `pair` matches each crossing A→B with a
  crossing B→A of the same name, otherwise the nearest by B's `seen`. Then:
  - `hop = ((landed_B - seen_A) + (landed_A - seen_B)) / 2`;
  - `offset = ((landed_B - seen_A) - (landed_A - seen_B)) / 2`;
  - the offset's drift is the slope of a linear regression of offset over `seen`.
  The pairing assumes the two directions' hops are equal.
- Reports, per direction: seen→first post, first→last post, last post→completion, the hop from seen, the hop from the
  last post, and the clock offset.

## 7. Configuration operands

**The bridge's operands.** The bridge has one operand: the canonical plist that `bin/mesh-user.sh` writes from the
consumer's JSON, TOML or YAML document through `bridge_configuration.py load`, to
`~/Library/Application Support/io.mesh.bridge/configuration.plist`. Nothing has a default. `configuration_read` reads
up to 64 KiB of it.

| key (document) | canonical | effect on the data flow |
|---|---|---|
| `purpose` | `purpose` | none (required, for people) |
| `window_gib` / `window_percent` (exactly one) | `window` (bytes; a percentage is resolved against the node's memory on the node that installs it) | the arena is `window / pgsz` pages = rows (`mesh_layout`); every lease, ring, cell, landing word, map record, lane page and relay lease comes from it; a window outside [1 page, 2^32 pages] is refused at start; every device registers it whole, and a device with fewer than `ceil(window / 1 GiB)` MRs refuses the port (`EFBIG`, with a 60 s to 30 min backoff) |
| `exec` (optional bool) | `exec` | `exec_served`: a `CHANNEL_OPEN` for `exec` spawns `$SHELL -c` (`child_spawn`); otherwise `EACCES` |
| `ledger` (optional bool) | `ledger` (written only if true) | `ledger_kept`: §6; up to 4.5 MiB per flow outside the window, plus a clock read per event on the progress threads |
| (the document's stem) | `name` | logged at start only |

Changing the plist takes effect only when the bridge restarts. `mesh-user.sh` re-copies the binary on a configuration
change, so the binary-change stop (§5) applies.

**Fixed in the source, not operands** (each changes the data flow):

| constant | value | where | what it changes |
|---|---|---|---|
| `MESH_PAUSE_NS` | 3 s | `mesh.h` | port turns, liveness turns, lanes retry, the 3 s timeout of a progress thread with nothing to look for |
| `MESH_PEER_NS` / `MESH_PAIR_NS` | 600 s / 30 s | `mesh.h` | flow, await and channel deadlines; the client's start bound (with `MESH_PAUSE_NS`) |
| `MESH_SILENT_NS` | 10 s | `mesh.h` | libnccl-mesh's silent-link cancel and drain bound only |
| `MESH_IDLE_NS` / `MESH_IDLE_POLL_NS` | 100 ms / 1 ms | `mesh-flow.c` | spin time before parking; poll period while parked |
| `MESH_PAIRS` / `MESH_LANES` / `MESH_FLOWS` / `MESH_HEADERS` | 16 / 2 / 256 / 8 | `mesh-flow.c` | max exclusive QPs a flow; lanes a port; flows a port; send-header frames a lane |
| `MESH_STRIPES` | 8 | `mesh-flow.c` | max stripes of a routed program |
| `MESH_PORTS` / `MESH_CLIENTS` / `MESH_DESTINATIONS` / `MESH_QUEUES` / `MESH_ORDERS` | 8 / 64 / 64 / 4 / 4096 | `mesh.h` | region tables |
| `MESH_UNSENT` | 64 MiB | `mesh-flow.c` | unsent control bytes before a link is dropped |
| `QD`, `MESH_REGION`, `MESH_BANK` | 4095, 1 GiB, 4 GiB | `mesh-verbs.h` | QP depth cap, MR extent, wire alias alignment |
| `MESH_LEDGER_EVENTS` / `MESH_LEDGER_LINES` | 65,536 / 2048 | `mesh-flow.c` | ledger bounds |

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
| How is a peer found and paired? | `port_turn`, `port_opened`, `hello_heard`, `port_greet`, `port_heard` |
| How is the map written and read? | `record_changed`, `record_heard`, `map_put`, `health_put`; `mesh-dataflow.c mesh_records_read`, `mesh_route` |
| What does each message do? | `port_heard`, `ready_heard`, `lanes_heard`, `channel_heard` |
| Exclusive or lanes? | `flow_make`, `flow_mode` |
| What happens at each flow state? | `flow_step`, `flow_done`, `flow_work`, `flows_turn` |
| How do routes, relays and stripes work? | `program_await`, `ready_heard`, `relay_direction`, `mirror`, `route_stripes`, `stripe_cable` |
| How does a publication become a SEND? | `mesh-collective.c mesh_host_publish`; `mesh.metal mesh_publish`; `mesh-flow.c flow_send`, `sends_configure` |
| When are receives posted? | `receives_configure`, `ring_advance` |
| Which word does a landing store? | `receive_input`; `flow_receive`, `lane_receive` |
| How does a relay hand a landing on? | `receive_input` (`release`), `send_wake` |
| How does the client see a landing? | `mesh-collective.c mesh_host_arrived`; `mesh.metal mesh_remote_wait`; `nccl-mesh.c landed`, `run` |
| How do lanes work? | `lane_send`, `lane_heard`, `lane_receive`, `lanes_turn`, `lanes_reset` |
| When do progress threads sleep or wake? | `port_progress`, `port_pass`, `port_list`, `port_unlist`, `mesh-call.c mesh_transfers_wake` |
| What happens on failure? | `flow_lost`, `flow_fail`, `flow_stop`, `flow_release`, `flow_free`, `port_lost` |
| Cancellation? | `mesh.h mesh_cancel`; `nccl-mesh.c cancel_all`, `progress_run` |
| ESTALE? | `program_stale`, `program_end`, `digest_apart`, `program_runs` |
| A client exits? | `clients_turn`, `flow_end`, `mesh-dataflow.c mesh_retire`, `mesh_retired_release` |
| Upgrade or stop? | `main` (the liveness block and teardown), `nccl-mesh.c ncclMeshBridgeStopping` |
| The ledger? | `ledger_open`, `ledger_seen`, `ledger_stamp`, `ledger_retired`, `ledger_landed`, `ledger_write`; `ledger.py` |
| Requests? | `asked`, `channel_heard`, `service_open`, `end_drain`, `requests_lost` |
| The QP budget? | `flow_make`, `lanes_turn`, `mesh-verbs.h pair_up`, `device_up` (`max_pairs`) |
| The configuration? | `bridge_configuration.py load`, `mesh-flow.c configuration_read`, `bin/mesh-user.sh` |

## Places where the code and bridge-and-ipc.md disagree

1. **Direct flows and the lanes.**
   - bridge-and-ipc.md#lanes (lines 452-454): "so a direct program's flows ride the lanes; the exclusive form stays
     in the code for a flow a READY asks it of".
   - The code: `program_take` asks every direct flow for exclusive QPs (`flow_make(..., 1)`, line 3101; commit
     921312ce). The paragraph above it in the same doc ("A program's operands ride exclusive flows alone") matches the
     code; this sentence is stale.
2. **The FAIL message's code.**
   - The #protocol table (line 218) says "a relay's goes on, its code kept".
   - `port_heard` fails the flow with `ECANCELED` unless the code is `ESTALE` (lines 2779-2781). `flow_fail` then
     tells the relay partner the local code (line 1538), so past a relay the original code is replaced by `ECANCELED`.
     Only `failed` (the node) and ESTALE survive.
3. **"A flow fails alone."**
   - #flows (line 381) says a flow fails alone.
   - `flow_fail` (lines 1539-1541) also fails every flow of the same owner and destination (the program's other
     cables and stripes; a relay lease's other stripes), and drops its await.
4. **Who writes in a client's slot and rows.**
   - #region (lines 103-104) and `mesh.h` say the bridge writes only the statuses, the ports and the map.
   - The bridge also writes into the client's leased rows: every SEND cell's `pair`, `span`, `request`, `successor`,
     `chunks`, `queue` and `port` (`sends_configure`), and every landing word (`flow_receive`, `lane_receive`).
5. **The ledger's "RC round trip".**
   - #ledger (line 556) and `ledger.py`'s docstring call last post→completion "the RC round trip".
   - Every QP is UC (`mesh-verbs.h pair_up`, line 202). Nothing in the code establishes what a UC SEND completion
     waits for on this provider.
6. **When RECORD is sent.**
   - The #protocol table's RECORD row (line 219) omits the periodic send.
   - `health_turn` (line 2073) sends this node's record on every port every 10 s and on urgent health changes
     (recovery.md#diagnosis describes this).

Outside bridge-and-ipc.md: heterogeneity.md §1 (line 28) says a link's beta is "8e9 over the widest paired port's
bandwidth". `mesh-dataflow.c mesh_paired` and `mesh-plan.h` use the slowest port.
