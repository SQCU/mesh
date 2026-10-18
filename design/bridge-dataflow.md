# The bridge's data flow

Where each byte and word of the mesh bridge goes: who writes it, who waits on it, and what happens when it fails. The
reasons for the design are in [bridge-and-ipc.md](bridge-and-ipc.md) (the bridge), [recovery.md](recovery.md) (loss
and recovery) and [heterogeneity.md](heterogeneity.md) (link costs and shares); this page repeats none of them.

Conventions. The page describes the code at mesh `f6432540` (`MESH_VERSION` 120). A file name with no directory is in
`rdma/`. A citation is `file function`. Line numbers are given only where the code is ambiguous. "Word" means an
8-byte atomic in the shared region. "Block" is `hdr.block * hdr.pgsz`. The bridge always starts with `block` = 1
(`mesh-flow.c main`: `mesh_layout(&geometry,pg,1,...)`), so a block is one 16 KiB page on Apple silicon, and a row is
one block.

## 1. The processes and memory

### Processes and threads

| process / thread | started by | does | waits on |
|---|---|---|---|
| `rdma/mesh-flow`, the bridge (one per node) | the launchd agent `io.mesh.bridge` (`bin/mesh-user.sh` writes it with `ProgramArguments` = mesh-flow and the canonical configuration plist, `KeepAlive`, `RunAtLoad`, stderr to `~/Library/Logs/io.mesh.bridge.log`); the keeper loads it at boot | everything below | — |
| the bridge's event loop (its main thread) | `mesh-flow.c main` | ports, control sockets, map, flow state machine, client slots, requests, health | `kevent64` with no timeout: `EVFILT_USER` (`wake`), sockets, `NOTE_EXIT` on client pids, signals, kernel network events, `EVFILT_VNODE` on its binary, the power sources' and thermal pressure's notify descriptors, and the configuration's `hello` and `health` intervals (`EVFILT_TIMER`) |
| `control_watch` thread | `main` | turns a change of `hdr.control` into a `wake()` of the event loop | `os_sync_wait_on_address(&M->control)` |
| port p's work queue `mesh.port` (serial dispatch queue) | `ports_found` | every verbs call except post and poll: open and retire the device, register the window, make, modify and destroy QPs and CQs, configure flow sides, write a flow's ledger | its jobs (`port_turn`, `flow_job`, `lanes_job`, `flow_stop`) |
| port p's two progress threads `mesh.rdma.receive` (side 0) and `mesh.rdma.send` (side 1), QoS user-interactive | `port_threads`, at the port's first listed flow or lanes connect; kept until `port_still` | side 1 polls SEND cells' `ready` words, posts SENDs and takes SEND completions; side 0 takes receive completions, stores landing words and reposts receives | spin while anything moves; parked on `hdr.bell[p]` (`port_progress`) |
| finder queue `mesh.ports` | `main` | reads the device list (`ports_find`) and the host name at each change configd notifies (`host_watch`, `host_changed`); takes the kernel's memory-pressure changes (a dispatch source: `pressured`, `wake`) | — |
| a client process (a rank, anything linked to libnccl-mesh or libmesh) | its user | attaches to the region, leases, binds, prepares, starts programs, publishes and consumes | its own landing words, `status[d].prepared` |
| a client's bridge watcher (one thread a process) | `mesh-dataflow.c mesh_watch`, at the first `mesh_open` of a bridge | at the bridge's exit: `bridge_pid = 0`, each slot of this process its words cancelled (`~0` where none landed, every one of a cyclic program's) and each destination's status `ESRCH` with its start woken, the map, health, control and stopping words woken | `kevent64`: `EVFILT_PROC` `NOTE_EXIT` on the bridge's pid |
| the client's GPU work | the client's command buffers | publishes into SEND cells; spins on landing words | `mesh.metal mesh_wait` / `mesh_remote_wait`, ended by a landing or a cancel |

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
rings_off     uint64_t[MESH_RINGS]          the stripe request rings' region offsets, 0 vacant (#stripes)
data_off      ARENA = the window: arena pages, registered on every port's device
```

Every time a client or reader maps the region, it checks the region's `magic` and `version` (`MESH_VERSION`) and checks that `bridge_pid` is alive
(`mesh-dataflow.c mesh_open`), and watches that bridge's exit (`mesh_watch`). A region of another version is refused
with `EINVAL`; a dead bridge gives `ESRCH`. A client that waits for a bridge to start blocks in `mesh_bridge_wait` on
the notify(3) name the bridge posts once its sockets listen (`MESH_STARTED`, a loop bridge's followed by its region's
name).

**Header words** (`mesh.h struct hdr`):

| word | written by | read / waited by | meaning |
|---|---|---|---|
| `magic`, `version`, `pgsz`, `block`, `rows`, `node`, offsets, `length` | bridge at start (`main`); `magic` last | clients at attach | geometry; `node` is a 31-bit fold of `gethostuuid` (waited for with no bound) |
| `bridge_pid` | bridge (start; 0 at stop); a client's watcher (0 at the bridge's exit) | every client wait (`mesh_gone`) | the bridge's process |
| `serial` | `mesh_slot_take` (clients and the bridge's own leases) | — | slot generation counter |
| `control` | clients (`mesh-call.c mesh_transfers_start`, `mesh-dataflow.c mesh_retire`); bridge (on a rebuilt binary, at stop) | the bridge's `control_watch` | "look at the client slots now" |
| `map`, `map_off`, `map_nodes` | bridge (`map_put`, `map_grow`) | `mesh-dataflow.c mesh_records_read` (seqlock, an odd value waited on untimed), `mesh_map_serial`, `mesh_map_wait` (untimed) | the map: an odd `map` value means a write is in progress |
| `health` | bridge (`health_put`) | `mesh_nodes_health` (a write overlapping a read: read again, an odd value waited on untimed) | a second seqlock for health-only record changes |
| `busy`, `idle_since` | bridge (`busy_tell`) | `mesh_busy`, `mesh-stat` | flows not failed or released plus awaiting programs; wall-clock ns at which the count last reached 0 |
| `stopping` | bridge (`main`, once its binary changed; it wakes the word's waiters) | clients (`mesh_stopping_wait`, an address wait; `nccl-mesh.c ncclMeshBridgeStopping`; metal-microbench `mesh_rank.m`) | "release held flows between executions" (§5) |
| `bell[8]` | clients (`mesh_transfers_wake`, `mesh_requests_wake`); bridge (`port_bell`, `send_wake`) | port p's progress threads, when parked | a futex-like word; this is not a hardware doorbell |
| `rings_off` table | clients (`mesh_requests_create`, `mesh_retire`); bridge (`stripes_requests_turn`, a gone owner's ring) | the bridge's event loop and send threads | the registered stripe request rings (#stripes) |

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

All QPs are `IBV_QPT_UC` (`mesh-verbs.h pair_up`), with MTU 4096, a GRH with hop limit 1 and the port's GID
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
 │ hdr: magic version node bridge_pid control map health busy stopping bell[8]      │
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
2. **Port turn**, where an event asked it (`port.wanted`: the port found, its link on or its interface attached, a
   paired link lost, a retirement, the configuration's `hello` interval while the port has no peer, a memory-pressure
   change after its registration was refused) and the port has no control socket (`port_turn` on the port's queue):
   `device_up` (context, port active, PD, window MRs), then `device_gid`. In `port_opened`: a registration refusal
   leaves the port closed until its next memory-pressure change or link event (`refused`); another failure with the
   device open retires the device and opens it again at once, and where that fails too, at the next event that asks
   its turn; on success the port records `max_pairs`, LID, GID and bandwidth, sets the phase to `MESH_PAIRING`, joins
   ff02::1 on the interface and sends a UDP `HELLO` (node, block bytes, port index) to `[ff02::1%enN]:18519`.
3. **Hello heard** (`hello_heard`). The higher node answers with its own HELLO. The lower node dials the higher's TCP
   listener at the address it heard from. Both ends `port_greet` (non-blocking, no SIGPIPE) and send a `HELLO` over
   the socket. The first message on a control socket must be a `HELLO` with an equal block size (`port_heard`),
   otherwise `EPROTO`. Once greeted, the bridge sets `link_set(..., MESH_PAIRED)` and `pairings++`, and sends every
   record in its map to the new peer. A link is lost by its events alone: its link off or interface detached, its
   socket's end of file or reset, a FAREWELL, a protocol error; a dial that never completes ends at the kernel's own
   connect error. A lost link that had paired says HELLO again at once (`port_lost`: `wanted`); one that never paired
   at its next event.
4. **A connection to the listener from loopback** is a client request (§2, Requests), not a peer.

### The map

- Each node has one record (`mesh.h struct mesh_record`): node, name, sequence, window, memory, model, the port
  table (`mesh_link_info[8]`: peer, phase, code, bandwidth, pairings, pairs, max_pairs) and health.
- `record_changed` builds this node's record with `sequence+1` and sends `RECORD` on every greeted port. It runs on
  any `link_set` that changes peer or phase, on a rename (`host_taken`), and from `health_turn`, which samples the
  node's health at each turn of the event loop: an urgent change (power, charge off AC, memory or thermal pressure,
  each also a notification the loop takes) is logged and sent at once, and the record is flooded again at the
  configuration's `health` interval (`EVFILT_TIMER`; none where it states none).
- `record_heard` keeps the newest record per node in `known[]`, apart from the map (RFC 2328 §13). A node's records
  are ordered by their sequence, the greater bytes newer among equal sequences (§13.1, `record_newer`). A record no
  newer than the newest known is dropped and not sent on; where the one known is newer, that one is sent back on the
  port it came from (§13 step 8). This node's own record from an earlier start of its bridge, newer than the one it
  holds, moves its sequence past it and sends its record again (§13.4), so a restarted bridge's records supersede its
  old ones by events alone. A newer record is taken and forwarded on every other greeted port. A copy of a pruned node
  is put back only where a path reaches that node again. A full `known[]` gives its place to the first entry whose
  node the map no longer holds.
- Writing the record:
  - If the links (each port's peer, phase and bandwidth), name, window, memory and model are unchanged
    (`record_links_same`), `health_put` writes it under the `health` seqlock, and the map serial does not move.
  - Otherwise `map_put` writes it under the `map` seqlock (odd while writing). `map_put` prunes nodes that no paired
    path reaches (BFS over `mesh_joined`), logs each pruned node with its last health, writes only when the map
    differs, and fails as values the programs awaiting a pruned node's READYs (`awaits_lost`: `EHOSTUNREACH` naming it).
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
other: `control_heard` raises `EPROTO`), `kind`, `node`, `count`, `key`, `code`, `block`, `port`, `from`, `to`,
`failed`, `bytes`, `lid`, `stripe`, `gid` and `pair[16][2]`. A payload of `bytes` follows the message. Sends queue in
the port's buffer behind what the socket has not taken (`port_say`, `port_flush`); a send that fails is the link's
loss.
A flow is named by `(port, key, near, far, stripe)` (`flow_on`); a message names it as `from` = the sender's end and
`to` = the far end.

| kind | sender, when | receiver's effect (`port_heard` unless noted) |
|---|---|---|
| HELLO | UDP: `port_opened` (each port turn while unpaired), `hello_heard` (higher node's answer); TCP: `port_greet` | greets the port (peer, peer port index, LID, GID) |
| RECORD | `record_changed`, `record_heard` (forwarding, or the newer known record sent back), a newly greeted port (every held record) | `record_heard` |
| READY (payload `struct ready`, route, transfer list) | `ready_say`: `program_take` (direct: both ends; routed: lower end, one per stripe), `program_await` and `ready_heard` (a higher end taking a held or awaited flow), a relay forwarding (`ready_heard`) | `ready_heard`: pairs an existing flow, holds a new flow, hands it to an awaiting program, refuses it (FAIL), or builds a relay (§3); one on a flow that heard a READY already is the peer's program made again: that flow fails (`ECONNRESET` naming the peer, tell 0) and the READY is taken as a new one |
| PAIRS | `flow_done` after `JOB_PAIR`: an exclusive flow's `qp_num` and PSN per QP; a lane rider's count 0 and `code = generation<<32 \| slot` | `HEARD_PAIRS`; stores the peer QPs or the peer's table index and generation; `flow_mode` ANDs the exclusive choice (`EPROTO` if the flow had already decided otherwise) |
| POSTED | `flow_tell`: once the flow's receives are posted (state `POSTING`); on a relay, also only after its partner heard the next hop's POSTED | `HEARD_POSTED`, which lets the flow run; on a relay, also tries `flow_tell(partner)` |
| FAIL | `flow_fail` with a `tell` on a flow not yet running, and `flow_end` of a flow not yet running whose program ended (a running flow's FAIL is a control frame, design/bridge-and-ipc.md#control); `ready_refuse` (a READY refused, as a value) | `flow_fail(ESTALE` if the code is ESTALE, else `ECANCELED`, domain 4); with ESTALE on a non-relay flow, also `program_stale` |
| LANES | `lanes_turn` once lanes are made (count 2, the peer QPs, `code` = header room); `lanes_reset` (count 0, `code` = why) | `lanes_heard`: stores the peer QPs and room, then connects; count 0, or new QPs while joined, resets this end's lanes (`lanes_reset`, untold) |
| FAREWELL | `main` at stop, with `code` = the signal, or -1 for a binary change; queued and flushed as far as the socket takes it, never waited on | `port.parted = 1` and `port_lost(ECONNRESET)` at once: every flow of the port lost, logged "parted" |
| STREAM | the sender's bridge once its stream's QP is made (`count` OFFER, its QP, `code` = the bytes); the receiver's once its receives are posted (POSTED, its QP) and once every word is checked (RESULT, `struct stream_result`); either end's failure (FAILED, `code` = why) | `stream_heard` (#streams) |
| STRIPES | `stripes_turn` once the stripe flow is made (count 2, the QPs, `code` = this end's largest piece); `stripes_reset` (count 0, `code` = why) | `stripes_heard`: stores the peer's QPs and piece, makes this end's (`wanted`), connects; count 0, or new QPs while joined, resets this end's (#stripes) |
| CHANNEL_OPEN / CONFIRM / REFUSE / DATA / WINDOW / SHUT / CLOSE | requests (below) | `channel_heard` |

### Client slots: `clients_turn`

`clients_turn` runs on every event-loop iteration; a client's change (its start, retire or wake) bumps `hdr.control`,
which `control_watch` turns into a `wake`.

1. A flow whose client owner is no longer owned or whose slot's `configured` is not that owner gets `flow_end(f)`: a
   slot retired in order (owner word `owner | MESH_RETIRING | MESH_ORDERLY`, `mesh_detach`) marks each running flow
   `FLOW_ENDING` (its send thread posts every cell the program released and takes those SENDs' completions, then says
   `drained`: §5); a flow not yet running, or any flow of an abort (the slot vacant or another's), has its program
   muted and fails, its peer told FAIL (a running flow's on its control channel, one not yet running's on the socket);
   a retire under way (`MESH_RETIRING` alone) is taken the next turn. A slot left `MESH_ORDERLY` is vacated the turn
   after the one that first saw it. Relay flows are excluded: their owner pid is the bridge's own.
2. For each slot: a new owner gets a kqueue `NOTE_EXIT` watch; if the pid is already gone (`ESRCH`), it is retired.
   In the event loop, `NOTE_EXIT` retires every slot of that pid (`mesh_retire`, an abort).
3. A slot with `configured == owner`, not yet in `accepted[s]`, and with no flow left from an earlier owner of the
   same slot, is accepted: `ports = 0`. If a held flow of its key names a different digest, `program_stale` runs.
   Otherwise `program_take(client, owner, d)` runs for each destination with transfers.
4. Once no flow references a non-owned owner (`zombie == 0`), `mesh_retired_release` returns closed rows to the
   window, and `mesh_reclaim` frees their arena pages.

**Retiring** (`mesh-dataflow.c mesh_retire_as`: `mesh_retire`, an abort, called by `nccl-mesh.c session_retire` for a
revoke or by the bridge; `mesh_detach`, in order, the client's close):

1. CAS `owner` to `owner | MESH_RETIRING`.
2. Set `configured = 0` and every list `length` to 0.
3. On every row the client owns, set `closed = 1` if it heads a lease, and clear its `ROW_OWN` bit.
4. Deregister the client's stripe request rings.
5. Set `owner = 0` (an abort: the slot vacant at once), or `owner | MESH_RETIRING | MESH_ORDERLY` (in order: the
   bridge vacates it once its flows have read it).
6. Wake every `status[d].prepared` waiter, so a pending start returns `ECANCELED`.
7. `mesh_notify(control)`.

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
| direct client flow (one per cable to the node, stripe j of n on cable j in `cable_rank` order) | `program_take` → `flow_make(..., exclusive=1)` | exclusive if both ends have room, otherwise the lanes | the client's |
| routed client flow (lower end; one per stripe of `route_stripes`, on `stripe_cable`) | `program_take` → `flow_make(..., 0)` | lanes | the client's |
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

Every state is left by an event alone: a message, a job's end, a failure or a revoke.

| state | entered | job run on the port's queue | leaves when |
|---|---|---|---|
| `READIED` | `flow_make` | — | READY or PAIRS heard → `JOB_PAIR` (`decided = 1`) |
| (`JOB_PAIR`) | | exclusive: `pair_up` (QPs in INIT, the data's and the control pair); lanes: nothing | done → send PAIRS → `PAIRING` |
| `PAIRING` | | | PAIRS heard, `!hold`, lanes up (or exclusive) → `JOB_RECEIVE` |
| (`JOB_RECEIVE`) | | exclusive: `pair_receive` (RTR), the control receives posted; `receives_configure` (exclusive: the first records posted by `ring_advance`); lane rider: `port_list(receiving)` | done → `POSTING`, `flow_tell` → POSTED |
| `POSTING` | | | POSTED heard, lanes up → `JOB_RUN` |
| (`JOB_RUN`) | | `sends_configure`; exclusive: `pair_send` (RTS); `port_list(progressing)` | done → `RUNNING`; `program_paired` once every flow of the program and destination runs and no await is left |
| `RUNNING` | | `JOB_CHANGE` if a relay side's lists arrive late: `port_quiet`, configure the side, list again | `flow_fail` / `flow_end`; a program ended in order once drained (`flow_released`); a relay once both ends said RELEASED (`relay_released`) |
| `FAILED` | `flow_fail`, `flow_end`, `flow_released`, `relay_released` | `flow_stop` (§5), at once or once its FAIL is told | `flow_free` once `stopped == 1` |

**Client status.** `flow_take` / `program_take` set `status[d].phase = MESH_PAIRING`. `program_paired` sets
`MESH_PAIRED`, `code = 0`, then `prepared = owner` and wakes the waiter. `status_failed` sets `MESH_STOPPED`, `code`,
`domain` and `failed`, then `prepared = owner`.

The client's start, `mesh_transfers_start`, waits on `prepared` for each destination with no bound
(`os_sync_wait_on_address`), woken by the bridge's status, by its own slot's retirement and by its watcher at the
bridge's exit. It returns:
- 0 once every destination is paired;
- the first stopped destination's `code`;
- `ECANCELED` if its own slot was retired (a revoke);
- `ESRCH` if the bridge is gone.

A peer whose program never starts is waited for until the caller revokes.

**Routed pairs** (`program_take`, `program_await`). The lower node routes on its own map and sends one READY per
stripe. The higher node awaits those READYs (a `struct await`, counted in `busy`). It takes held flows at once and the
rest as `ready_heard` delivers them; each take answers with a READY back along the relays. Where its own map holds no
route to the lower node the program fails `EHOSTUNREACH` at once, as the lower's fails where its map holds none to the
higher; otherwise the await lasts until the READYs come, the lower's failure, a revoke of its program, or the map's
prune of the lower node (`awaits_lost`: `EHOSTUNREACH` naming it).

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
   the ports bit and the status; the flow then runs. It is held until a program takes it, the sender's FAIL (its
   program ended or revoked), or the link's loss.
2. A client keeping its sessions, and so their RUNNING flows, past an execution (metal-microbench
   docs/placement.md#retention-and-protention-the-next-execution-planned-ahead). The bridge sees ordinary RUNNING
   flows, counted in `busy`; `stopping` asks the client to release them (§5).

**`flow_free`** (event loop, once `flow_stop` reported `stopped == 1` and no job runs):
- unlinks the flow and its partner;
- `port.flows--`;
- frees the table index (the next `flow_make` at that index bumps its generation, so a lane header naming the old
  generation is discarded);
- `committed -= qps + 1` if the flow was exclusive (its control pair too), and its control page returned;
- `port.frees++` (a flow whose release was refused tries again);
- drops both sides' leases (a relay lease is retired once its last user drops it: `lease_free` → `mesh_retire`, and
  `unreleased = 1`);
- `flow_count--`, then `busy_tell`.

### Progress threads (`port_progress`, `port_pass`)

**Passes.**
- One pass of side 1 runs `flow_send` on every listed, progressing exclusive flow (its streams, none of a muted
  program's; then its control frames, `flow_control`: a FAIL the event loop asked told (`FLOW_TELLING`, `said` once its
  completion is taken), else where the configuration states `keepalive` an ALIVE frame once that long passed,
  design/bridge-and-ipc.md#control; then its SEND completions, every one counted taken; then, for a program ended in
  order, `drained` once every released cell is posted and every SEND completed); then, while a lane rider is listed, a
  lane SEND is in flight or a control entry of the port's is queued, `lane_send` for both lanes (lane 0's header also
  carrying the port's queued FAILs and RELEASEDs, `port_tell`), `lane_retire`, and `lanes_drained` (a rider of a
  program ended in order `drained` once its streams are idle and the lanes' gates retired past its `lane_mark`).
- One pass of side 0 runs `flow_receive` on every listed exclusive flow (a landing storing its word, a control frame
  applied and its receive posted again, `frame_heard`), then `lane_receive` while a lane rider is listed or a lane's
  records are posted and not landed.

**Parking.**
- A thread with something to look for (`any`: a completion queue that can still return something, which no event of
  Apple's provider announces) that moves nothing for `MESH_IDLE_NS` (100 ms; counted in `MESH_IDLE_PASSES` =
  4096-pass batches) parks on `bell[p]` and looks again every `MESH_IDLE_POLL_NS` (1 ms).
- With nothing to look for, it parks on `bell[p]` with no timeout.
- A bell ring wakes it immediately: a client's group starting (`mesh_transfers_wake`), a landing that released one of
  the port's SEND cells while its send thread was parked (`send_wake`), a flow listed or unlisted, a control entry
  queued, the lanes', a stripe flow's or a stream's change, the guard, the bridge stopping. Before it sleeps, a thread
  that told it is parked passes once more and reads the guard and `closing` after the bell's value it waits on, so a
  ring between its loop's head and its sleep is never slept past.

**Table changes.** Flows enter the port's table with `port_list` (release store, then a bell ring). They leave with
`port_unlist` and `port_settle`: `epoch++`, a bell ring, then an address wait on each thread's `seen` until it is at
least `epoch` (`port_told`: the thread stores `seen` and wakes its waiter). After that, no thread reads the flow. A
thread stores `seen` only when the epoch it read differs from the one it last told, so the line of `high`, `epoch`
and `seen` stays clean in both cores between table changes.

## 4. The life of one transfer

This section follows one libnccl-mesh large-class position `at` from rank A (node a) to rank B (node b) over a direct,
exclusive flow. Each step names the word written, its writer and its reader. The constants:
- `LAG = MESH_LAG = 8`, `DEPTH = 16` (ring slots), `POSITIONS = 32` (invocations of the cyclic program;
  `nccl-mesh.c`).
- Position `at` is invocation `t = at % 32` in cycle `c = at / 32`, and uses ring slot `at % 16`.
- The large slot is `NCCL_BUFFSIZE / 8` rounded to a block (512 KiB, `mesh-collective.c mesh_wire_of`), which is 32
  chunks of 16 KiB and 128 frames of 4 KiB. The small slot is the communicator's `small` (one block where none is
  given; the pair's decode passes 3,072 bytes, one frame), on a ring whose slots start a block apart (the transfer's
  `invocation_pages`, its `bytes` the slot's). The large class rides queue 0 and the small class queue 1
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
   `ready > stream->cycle`. A parked thread looks again within `MESH_IDLE_POLL_NS` (1 ms) while the flow is listed, or
   at once when the group's start rang the bell (`mesh_transfers_wake`, called by `nccl-mesh.c session_run`, by the Metal issue path,
   and by `ncclMeshComplete`). `mesh_host_publish` itself rings no bell.
5. **Posting.** For each chunk: if the gate has room (frames in flight `posted - retired` plus this request's
   `ceil(len/4096)` frames, at most `gate.capacity` = `min(send_capacity, sent CQ cqe / qps)`), post a copy of the
   prepared request with `IBV_SEND_SIGNALED` and `wr_id = queue << 48 | posted frames`. Do this on QP `cell->queue`,
   one request per chunk. The cell's own `request.wr_id` stays 1, the producer's argument. When the chain is done,
   `stream->cell = successor`; on reaching the first cell again, `stream->cycle++`.
6. **SEND completion.** `send_retire` polls the shared `sent` CQ, counts every completion taken
   (`provider.sent_polled`, which the teardown's flush reads) and sets `gate.retired` from the `wr_id`. An error status
   raises `flow_lost(status, domain 2)`. The device completes every SEND (`mesh-verbs.h struct mesh_queue`).
7. **Hardware credit.** TN3205's credit flow control holds each SEND until the peer has posted its matching receive
   (`sends_configure`). The bridge has no software credit protocol on an exclusive flow.

**Receiving (node b):**

8. **Receives posted ahead.** `ring_advance` posts record n once record `n - behind` has landed. `behind` spans the
   transfer's ring depth less one, divided by the stripes; for this session on one cable that is 15 invocations.
   Posting also
   stays within `capacity` = `min(receive_capacity, completion CQ cqe / qps)` frames. Gating uses landings and
   capacity alone (`ring_advance`). Nothing in the bridge waits for the local reader to consume a slot, or for a
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
      until `word >= 1 + c` (`mesh.metal`). `~0` also satisfies the wait, and the caller tells the two apart: a cancel
      read is recorded in the session's violation words [2] and [3], and every later wait of the program ends there.
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

## 5. Failure, cancellation and release

Every failure here is an event: an error completion or post, a protocol error, a peer's FAIL or FAREWELL, the link
off or its interface detached, a control socket's end of file or reset, a client process's exit, a revoke, the map's
prune. Nothing in the bridge or libmesh ends a wait by a clock.

### The `~0` landing word

The bridge cancels a program's words only past each ring's landed count (`mesh-flow.c flow_cancel`,
design/bridge-and-ipc.md#control): from the ring's first record not landed, through the program's invocations less the
ring's least depth (at least one), every record not landed in a one-pass program; landed words stay. A flow whose
receives were never prepared lands nothing: every word toward its node. Every spin in `mesh.metal` and every host check
treats `~0` as "cancelled"; a GPU walk's wait that reads it records it in the session's violation words, and every
later wait of the program ends (nccl-mesh-metal.m `mesh_step_check`), so a cancel of the positions past the landed
count ends every wait. It happens in `flow_stop(code != 0)` while the owner is still owned, after `port_unlist` settled
the port's progress threads (no landing of the flow is stored after it). The library cancels nothing itself; at the
bridge's exit a client's watcher (`mesh-dataflow.c mesh_watch`) cancels its own slots' words, `~0` where none landed
and every one of a cyclic program's, so no wait of the process outlives its bridge.

**A late lane landing.** `lane_receive` stores a landed record's word through `record->link`, whether or not that flow
is still in the port's table, so a lane rider's records already posted can land after the cancel. Each is a record past
the landed count at the cancel, and stores its own position's word: a reader waiting there sees a landing that came,
and no reader waits on a later position of that word before passing it (in order; a GPU walk halts at the first
cancel). `flow_release` waits for those frames (`lane_frames`) before it frees anything; the last one's landing wakes
the event loop where the flow's release is under way (`FLOW_RELEASING`).

### `flow_fail` and `flow_stop`

`flow_fail(f, code, domain, tell, where, origin)` (event loop):

1. A client program's flow: every flow of the program is muted first (`program_quiet`: `ending |= FLOW_MUTED` on each,
   its ports' progress threads settled), so no cell of it is posted past this.
2. `state = FAILED`, the failure logged.
3. If `tell` is set: a running exclusive flow's FAIL (`code`, the node `origin`) is told by its send thread on its
   control queue pair (`control_tell`: `FLOW_TELLING`; the flow stops once that FAIL's completion is taken, `said`, or
   the peer's own FAIL came, or its link is lost: `flows_turn`, `port_lost`); a running lane rider's is queued for lane
   0's next header (`port_tell`) and the flow stops at once; a flow not yet running's goes on the control socket and
   the flow stops at once.
4. `flow_stop`, at once or once told.
5. Fail the relay partner, telling FAIL.
6. Fail every other flow of the same client program (toward every node, its other cables and stripes) with FAIL; a
   relay lease's, its other stripes.
7. Drop the program's await.

A data thread's failure is `flow_lost`: CAS `failed`, `ending |= FLOW_MUTED`, `wake`; the threads go on serving the
flow, so its send thread can tell FAIL. A peer's FAIL heard (`control_apply`) sets `heard_fail`, mutes nothing more
and stops the threads' serving; `flows_turn` fails the flow, telling nothing back.

`flow_stop`, on the port's queue:

1. `port_unlist`: the progress threads stop reading the flow.
2. If `code != 0` and the owner is still owned: `flow_cancel` (above), then `status_failed` (phase `MESH_STOPPED`,
   `code`, `domain`, `failed = where`, `prepared` woken).
3. `flow_release`: wait for the lane frames (0 returned while a rider's records are posted on lanes not yet torn
   down), then under the port's guard `pair_flush` (each queue pair to the error state, both completion queues drained
   of every work request posted and not yet taken: "flushed N of M work requests in T us") and `down_pair` (-1 where
   a destruction is refused, its handles kept), then the census (each ring's requests, frames, completions, records
   posted and landed, cells sent; the control frames and keepalives) and `ledger_write`, and the rings and streams
   freed.
4. Set `stopped` to 1 (released: `wake`), 2 (records still posted: the last one's landing wakes the event loop, and
   the release is tried again) or 3 (a destruction refused: tried again once another flow of the port is freed or the
   link is lost, `port.frees`).

### A release in order

A program that ended in order (`mesh_detach`: ncclCommDestroy) tells its peers nothing.
1. `clients_turn` → `flow_end`: each running flow `ending |= FLOW_ENDING`, its bell rung.
2. Its send thread posts every cell the program released (its streams idle: each next cell not released), takes those
   SENDs' completions (an exclusive flow's gates `requests == completions` and no control frame in flight; a lane
   rider's lanes' gates retired past its `lane_mark`), and says `drained` (`flow_drained`: `wake`).
3. `flows_turn` → `flow_released`: `state = FAILED`, "ended in order" logged; a lane rider queues RELEASED for its
   peer (`port_tell`); `flow_stop(0)`: no cancel, no status, the flush and release above.

A peer's flow is released by its own client's detach, or its program's failure, or its link's loss. A relay has no
client: a relay flow that hears RELEASED from its end passes it on by its partner (`relay_released`), and the relay's
two flows are released (`flow_stop(0)`, "released by both its ends") once each end's RELEASED came. Both ends'
RELEASED mean each end had every cell it waited on and sent every cell it released, so nothing the relay carries is
still wanted. A relay whose end fails hears its FAIL or its link's loss, and fails both flows.

Domains: 1 = a post, ring or job error (`flow_lost(...,1)`, `flow_done`, lanes); 2 = an error completion status; 3 = a
poll error; 4 = a protocol or program-level failure.

### How each failure reaches the clients

| event | bridge path | what the local client sees | what the peer side sees |
|---|---|---|---|
| a data error on a flow (post, poll or completion) | progress thread `flow_lost` (CAS `failed`, the program muted, `wake`), then `flows_turn` → `flow_fail(..., FAIL)` | its words past their landed counts `~0`, every destination's; status `MESH_STOPPED`, code and domain, `failed` = this node | FAIL → `ECANCELED`; its words past landed `~0`; status naming this node |
| the peer's program ended in order (`mesh_detach`, ncclCommDestroy) | peer: `flow_end` → drained → `flow_released`: its flows released, nothing told; a lane rider's RELEASED for the relays | nothing while the peer makes no program of the execution again; where it does, its READY fails this end's flow (`ECONNRESET` naming the peer, its words past their landed counts `~0`), and a session on it reopens at its next group | — |
| the peer's program aborted (exit, revoke, its failure) | peer: `flow_end` / `flow_fail`: its program muted, FAIL on each running flow's control channel | its words past their landed counts `~0`, status `ECANCELED` naming the peer; its own program's other flows fail (FAIL) | — |
| digests differ (ESTALE) | `ready_heard` / `clients_turn`: `ready_refuse(ESTALE)` or `flow_fail(ESTALE)`, then `program_stale` → `program_end` for every destination with transfers | every destination `ESTALE`, naming the node whose program differs; start fails (`start_failure`) | FAIL with ESTALE → `program_stale` there too (not for relays) |
| maps disagree on a cable or route; READY shape mismatch | `ready_heard` / `program_take` / `program_await`: `ready_refuse(EPROTO)` + `program_fail(EPROTO)` | `EPROTO` naming the refusing node | FAIL → `ECANCELED` |
| no route to a routed peer; the map prunes an awaited node | `program_take` / `program_await`; `map_put` → `awaits_lost` | `EHOSTUNREACH` naming that node | — |
| a table or device full | `flow_make` NULL → `program_fail(E2BIG)` / `ready_refuse(E2BIG)`; `pair_up` `ENOSPC` → `flow_done` → `flow_fail(ENOSPC, 1)` | `E2BIG` / `ENOSPC` | FAIL |
| a link lost (socket EOF or error, send failure, `EPROTO`, a refused dial, `KEV_DL_LINK_OFF` / `IF_DETACHED` → `ENETDOWN` with retire) | `port_lost`: close the socket; `lanes.up = 0`; `flow_fail(code` (`ECONNRESET` if parted), `4, tell 0, where = peer)` on every flow of the port, and a flow still telling its FAIL stopped; `requests_lost`; lanes and stripes CLOSING; `link_set(PAIRING` or `STOPPED)` → RECORD → map serial; a paired link says HELLO again at once; on retire, `down_device` at a later `port_turn` once no flow or lane holds QPs | words `~0`; status naming the peer; libnccl-mesh fails `ncclRemoteError` ("the link to node N was lost"); the next session reads the new map | the same, on the other end |
| lanes fail (completion error, header out of step, made or run) | `lanes_lost` → `lanes_turn` → `lanes_reset`: LANES of count 0 to the peer; `flow_fail(code, 1)` for every lane rider at `POSTING` or later or in a receive/run/change job; the lanes closed and made again at once; the link and its exclusive flows go on. Lanes not connected: the link lost and its device retired (`port_lost(code, 1)`), opened again | lane riders' words `~0` | LANES of 0 → its lanes reset, and its riders fail |
| the peer's bridge stops | peer: FAREWELL, then close; here: `parted = 1` and `port_lost(ECONNRESET)` at the FAREWELL | `ECONNRESET`, logged "parted" | — |
| this bridge stops (signal, or a changed binary with `busy == 0`) | `main`: `requests_stop`; per port FAREWELL (flushed as far as the socket takes it), `parted = 1`, `port_lost(ESHUTDOWN, retire)`; on each port's queue `flow_release`, then `port_still`; the stripe flow's and a stream's queue pairs flushed and destroyed, `down_pair(lanes)`, `down_device`; `bridge_pid = 0`; `mesh_notify(control)`; `atexit` `shm_unlink` | flows that remain fail with status `ECONNRESET`, `failed` = the peer node. Later waits see `bridge_pid` 0 (`ESRCH`); a crash's exit is the clients' watchers' event (`mesh_watch`: their words cancelled, statuses `ESRCH`). A lease of the old region fails the next libnccl-mesh lease or session (`ncclMeshMemLive`, `attaches_stale`) | FAREWELL → `ECONNRESET` |
| a client process dies | `NOTE_EXIT` → `mesh_retire` (an abort) → next turn `flow_end`: the program muted, FAIL on each running flow | — | `ECANCELED`; on the loopback fabric its peers' calls failed 1.1-2.5 ms after the kill (rdma/mesh-loop.c `kill`) |

**The `stopping` word.**
1. On an `EVFILT_VNODE` event of its binary (its contents written, or the file replaced, renamed or removed; an
   attribute's change, such as an extended attribute the system adds, is none), `main` sets `hdr.stopping = 1`, wakes
   its waiters and calls `mesh_notify(control)`. `bin/mesh-user.sh` renames a new binary into place, and copies it anew
   when only the configuration changed, so a configuration change also fires this.
2. A client holding sessions between executions reads it (`mesh_stopping_wait`, an address wait; or
   `ncclMeshBridgeStopping`, metal-microbench `mesh_rank.m` `release_held`), destroys its communicators, and holds none
   afterwards.
3. Once `busy` is 0 (no flow not failed or released, no awaiting program), the bridge stops
   (`stop_reason = MESH_STOP_BINARY`) and launchd starts the new binary.
4. A signal stop never sets `stopping`.

## 6. The ledger

**Switch.** The configuration's `ledger = true` sets `ledger_kept` (`configuration_read`). The start log line then
says "ledgers kept".

**What is recorded, per flow.** `ledger_open` runs on the port's queue at the flow's first non-pair job, before
either side is configured.

| event | recorded by | fields (each time is `ledger_ns` = `CLOCK_UPTIME_RAW` ns of that node, the clock the rank reads) |
|---|---|---|
| SEND cell first seen released | `flow_send` / `lane_send` → `ledger_seen` | the cell, the stream cycle, `seen` |
| its SENDs posted | `ledger_stamp` | `first` (the first post), `posted` and `mark` (the gate's frames after the last post; an exclusive flow only) |
| its last SEND completed | `send_retire` → `ledger_retired` | `completed`, taken when the queue's retired frames reach `mark` (completions are FIFO per QP); a lane rider has none (0) |
| a record landed | `flow_receive` / `lane_receive` → `ledger_landed` | the record index, cycle, `landed` (after its word was stored) |

**Names.** Cells and records are named at configure time (`sends_configure`, `receives_configure`) by queue (ring),
invocation, position among that invocation's transfers on the queue, binding, chunks and chunk; a cell also keeps its
transfer's bytes (`out[i].bytes`). Both ends' lists are in the same order, so a cell and its records share (queue,
invocation, position, cycle) across the link.

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
- Event lines: `{"s":[id,queue,invocation,position,binding,cycle,chunks,seen,first,posted,completed,bytes]}` and
  `{"r":[id,ring,invocation,position,binding,chunk,chunks,cycle,landed]}`. A SEND line's `bytes` (last, after the
  fields it had before) is its cell's transfer's, -1 where the cell was not named; metal-microbench
  `tools/torch_parallel/timeline.py` takes the wire's time out of a crossing's by it.

**Reader, `ledger.py`:**
- `read`: parses JSON from the log lines (dropping the `say` prefix), follows a header's `file` beside the log or in
  any directory given, and attaches events to the latest header of their ledger id.
- `peer_of`: the same key and stripe, with node/peer and near/far swapped; ties are broken by shared names.
- `crossings`: a cell counts only when all its chunks landed, at the last chunk's time, with its SEND's fields
  (`bytes` among them where the ledger recorded it); every invocation's crossings are read.
- Clock offset (each node's own `CLOCK_UPTIME_RAW`, with an unknown offset): `pair` matches each crossing A→B with a
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
| `window_gib` / `window_percent` (exactly one) | `window` (bytes; a percentage is resolved against the node's memory on the node that installs it) | the arena is `window / pgsz` pages = rows (`mesh_layout`); every lease, ring, cell, landing word, map record, lane page and relay lease comes from it; a window outside [1 page, 2^32 pages] is refused at start; every device registers it whole, and a device with fewer than `ceil(window / 1 GiB)` MRs refuses the port (`EFBIG`; the port turns again at its next memory-pressure change or link event) |
| `exec` (optional bool) | `exec` | `exec_served`: a `CHANNEL_OPEN` for `exec` spawns `$SHELL -c` (`child_spawn`); otherwise `EACCES` |
| `ledger` (optional bool) | `ledger` (written only if true) | `ledger_kept`: §6; up to 4.5 MiB per flow outside the window, plus a clock read per event on the progress threads |
| `keepalive_us` (optional) | `keepalive` (ns) | an exclusive flow's send thread posts an ALIVE frame on its control queue pair once that long passed since its last frame (`flow_control`); none where unstated |
| `hello_us` (optional) | `hello` (ns) | the event loop's `HELLO_TIMER`: each port with no peer, no socket and no refused registration says HELLO again; none where unstated |
| `health_us` (optional) | `health` (ns) | the event loop's health timer: this node's record flooded again (`health_turn`); none where unstated |
| (the document's stem) | `name` | logged at start only |

Changing the plist takes effect only when the bridge restarts. `mesh-user.sh` re-copies the binary on a configuration
change, so the binary-change stop (§5) applies.

**Fixed in the source, not operands** (each changes the data flow):

| constant | value | where | what it changes |
|---|---|---|---|
| `CONTROL_SENDS` / `CONTROL_RECEIVES` / `CONTROL_SLOT` | 64 / 192 / 64 B | `mesh-flow.c` | an exclusive flow's control frames in flight and receives posted, a receive slot (design/bridge-and-ipc.md#control) |
| `MESH_TELLS` | 256 | `mesh-flow.c` | a port's queued control entries of its lane riders (FAIL, RELEASED) |
| `MESH_IDLE_NS` / `MESH_IDLE_POLL_NS` | 100 ms / 1 ms | `mesh-flow.c` | spin time before parking; look period while parked with a completion queue to poll (none: the bell alone) |
| `MESH_PAIRS` / `MESH_LANES` / `MESH_FLOWS` / `MESH_HEADERS` | 16 / 2 / 256 / 8 | `mesh-flow.c` | max exclusive QPs a flow; lanes a port; flows a port; send-header frames a lane |
| `MESH_STRIPES` | 8 | `mesh-flow.c` | max stripes of a routed program |
| `MESH_PORTS` / `MESH_CLIENTS` / `MESH_DESTINATIONS` / `MESH_QUEUES` / `MESH_ORDERS` | 8 / 64 / 64 / 4 / 4096 | `mesh.h` | region tables |
| `QD`, `MESH_REGION`, `MESH_BANK` | 4095, 1 GiB, 4 GiB | `mesh-verbs.h` | QP depth cap, MR extent, wire alias alignment |
| `MESH_LEDGER_EVENTS` / `MESH_LEDGER_LINES` | 65,536 / 2048 | `mesh-flow.c` | ledger bounds |

**Operands of the clients that shape the bridge's traffic:**
- `NCCL_BUFFSIZE` (environment, default `MESH_BUFFSIZE` 4 MiB; `nccl-mesh.c setting`) sets the large slot through
  `mesh_wire_of`.
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
| When do progress threads sleep or wake? | `port_progress`, `port_pass`, `port_told`, `port_list`, `port_unlist`, `port_settle`, `port_guard`, `mesh-call.c mesh_transfers_wake` |
| What happens on failure? | `flow_lost`, `flow_fail`, `control_tell`, `port_tell`, `flow_stop`, `flow_release`, `flow_free`, `port_lost` |
| Cancellation? The release in order? | `mesh-flow.c flow_cancel`, `program_quiet`, `flow_stop`; `flow_end`, `flow_drained`, `lanes_drained`, `flow_released`, `relay_released`; `mesh-dataflow.c mesh_watch` |
| ESTALE? | `program_stale`, `program_end`, `digest_apart`, `program_runs` |
| A client exits? | `clients_turn`, `flow_end`, `mesh-dataflow.c mesh_retire`, `mesh_retired_release` |
| Upgrade or stop? | `main` (its `EVFILT_VNODE` and teardown), `mesh-dataflow.c mesh_stopping_wait`, `mesh_bridge_wait`, `nccl-mesh.c ncclMeshBridgeStopping` |
| The ledger? | `ledger_open`, `ledger_seen`, `ledger_stamp`, `ledger_retired`, `ledger_landed`, `ledger_write`; `ledger.py` |
| Requests? | `asked`, `channel_heard`, `service_open`, `end_drain`, `requests_lost` |
| The QP budget? | `flow_make`, `lanes_turn`, `mesh-verbs.h pair_up`, `device_up` (`max_pairs`) |
| The configuration? | `bridge_configuration.py load`, `mesh-flow.c configuration_read`, `bin/mesh-user.sh` |

## Stripes

The operator, 2026-10-01 (metal-microbench docs/principles.md section 2, The bridge): "MAKING K-MANY COMPUTERS INTO A
SHARED ADDRESS SPACE WHERE EXTREMELY HIGH NUMBERS OF CPU CORES CAN ASYNCHRONOUSLY START WRITING TO STRIPES OF THE REALLY
BIG SHARED ADDRESS SPACE"; and the same day, of the bridge, an API "which lets ANY PROGRAM IN THE UNIVERSE to say 'HEY,
I WOULD LIKE TO WRITE TO A GLOBALLY ACCESSIBLE WORKSPACE ANY PROGRAM IN THE UNIVERSE ON THE OTHER COMPUTER CAN READ,
YAH/NAH?' then gets a 'YAH, I WROTE THAT'". Stripe writes landed in mesh 8fc27cbe (MESH_VERSION 120 of that line, 15:07)
and went 19 minutes later with eight other commits in 753676bc, for a double deregistration and two kernel panics
that were never located (design/RDMA-KERNEL-RECOVERY.md "The stripe writes of October 1"). They are restored here on
the resident bridge at MESH_VERSION 118: any thread of any client process, or a GPU, writes a stripe of its node's
window into the same offset of a neighbour's window, and no program runs at the peer.

**The request** (`mesh.h struct mesh_requests`, `struct mesh_stripe`; `mesh-call.h`):

| step | who | what |
|---|---|---|
| a ring | the client, `mesh_requests_create(context, entries, &ring)` | a lease of its slot (`mesh_section_create`) of 128 bytes and `entries` × 64; `owner` its slot's owner word, `at` its region offset; its offset compare-exchanged into a vacant word of the region's ring table (`hdr.rings_off`, `MESH_RINGS` = 256 words; `EBUSY` where none is vacant) |
| exact pages | the client, `mesh_section_at(context, page, bytes, &section)` | a lease at exactly arena page `page` (8fc27cbe's), so both nodes' clients place what they exchange at one offset; `ENOMEM` where a page is held (`mesh-dataflow.c mesh_arena_claim`) |
| a request | any thread, `mesh_write(ring, node, offset, bytes, word, value)` | `head` taken by compare-exchange (`EAGAIN` while `entries` requests are untaken), the entry written, `ready = k + 1` stored last; `offset` and `bytes` a range of the window, `word` 0 or an 8-byte-aligned region offset in the window (`EINVAL` otherwise). A GPU writes the entry and `ready` itself |
| the wake | `mesh_requests_wake(ring, node)` (`mesh_write` calls it) | the bell of each port paired with `node` (its send thread takes the request at once), and `hdr.control` (the event loop makes the stripe flow where none is up) |
| deregistration | `mesh_retire` (the client's detach, or the bridge for a dead pid) | each table word naming a ring of that owner set to 0; the bridge also drops a ring whose owner is not owned (`stripes_requests_turn`) |

**The event loop** (`stripes_requests_turn`, every iteration): for each registered ring, the request at `taken`, where
`ready == taken + 1`:
- toward a node one of whose cables has its stripe flow up: left to the send threads;
- toward a node with cables and no stripe flow up: each of those ports `wanted`, and the request waits while one is made;
- refused as a value (`taken` compare-exchanged, the writer's `word` set to `UINT64_MAX`, the ring's `failed` + 1 and
  `code`), and logged once per code: toward this node (`EINVAL`), toward a node with no cable (`EHOSTUNREACH`: a stripe
  rides one link; routed stripes are not built), or where every cable's stripe flow last failed to be made or connected
  (its `code`: `ENOSPC` the device's queue pairs, `ENOMEM` the descriptor pages, a connection's error); that request
  clears those codes, so the next one makes the stripe flow again.

**The stripe flow** of a port (`struct stripes`, `stripes_turn`, `stripes_job`, `stripes_heard`), made as the lanes are
and kept while the link is paired:

| state | entered | job (port's queue, always guarded: below) | leaves when |
|---|---|---|---|
| `STRIPES_DOWN` | start; a closing done | — | `wanted` (a request, or the peer's STRIPES, which clears a standing code), the link greeted, no failure's `code` standing, its descriptor pages leased (`2 × MESH_DESCRIPTORS × 64` bytes, once) and two queue pairs of room (`committed + 2 <= max_pairs`) → MAKING; a make it cannot start sets `code` |
| `STRIPES_MAKING` | | `pair_up(2)`: control pair 0, data pair 1, a receive and a SEND completion queue | done → `STRIPES` (count 2, the two QPs and PSNs, `code` = this end's largest piece) → MADE |
| `STRIPES_MADE` | | — | the peer's STRIPES heard → CONNECTING |
| `STRIPES_CONNECTING` | | `pair_receive`, `pair_send`, `stripes_ready` (piece = the lesser of the two ends', the data pair's frames less the descriptors their completion queues hold, every count 0, the 256 descriptor slots' receives posted on pair 0), `port_threads` | done → UP (`up = 1`, the bell rung) |
| `STRIPES_UP` | | the progress threads serve it at every pass (`port_pass`) | a failure (`stripes_lost` → `stripes_reset`, STRIPES of 0 to the peer; a make or connect that failed leaves its `code` standing), the peer's STRIPES of 0 or of new QPs, the link lost → CLOSING |
| `STRIPES_CLOSING` | | `up = 0`, `port_settle` (neither thread reads it), `stripes_abandon` (each request taken and not completed fails: `ECONNRESET`), `pair_flush` (each QP to `IBV_QPS_ERR`, both completion queues drained of every work request posted and not taken), `down_pair` | done → DOWN, `committed -= 2`, "stripes closed (N completions flushed)" logged |

A piece is at most `MESH_STRIPE_PIECE` (4 MiB, 8fc27cbe's `MESH_STRIPE_CHUNK`) and the device's `max_msg_sz` in whole
frames, the lesser of the two ends' (STRIPES' `code`), and never crosses a 1 GiB registration (`stripe_length`), so both
ends cut a stripe alike.

**Sending** (`stripes_send`, the port's send thread, while `up`):
1. SEND completions from the `sent` queue: a descriptor's frees a descriptor credit (`descriptors_done`); a piece's
   frees its frames and its slot of the piece ring (`MESH_STRIPE_CHUNKS` = 1024); a stripe's last piece's stores
   `value` into the writer's `word` (where the writer's slot is still owned), `written + 1`, `flight - 1`.
2. With no stripe being cut and fewer than 256 descriptors in flight: the rings from `cursor`; a ring's request at
   `taken` that is ready and names this port's peer is read, then taken by compare-exchange of `taken` (every port
   toward the node scans every ring; each request is taken once, by the first), `flight + 1`. A range past this
   window or the peer's (its record's `window`) or a word that is no word fails as a value (`EINVAL`).
3. The descriptor (`offset`, `bytes`, `word`, `value`; 64 bytes) copied into a sent slot of the descriptor pages and
   posted as one SEND on pair 0.
4. The pieces posted in order on pair 1 while the pair's frames (`send_capacity`) and the piece ring hold them; a
   stripe of 0 bytes is one SEND of 0 bytes, so its word still lands.

**Receiving** (`stripes_receive`, the port's receive thread, while `up`):
1. Receive completions (an acquire fence after a poll that took any): a descriptor's slot appended to the arrival
   FIFO (`arrived`); a piece's frees its frames and slot, and a stripe's last piece's stores `value` into `word` in
   this node's region, `landed + 1`.
2. The FIFO's head descriptor read from its slot (a range past this window or a word that is no word is the peer's
   protocol error: `EPROTO`, the flow reset); its pieces' receives posted at its offset, cut as the sender cuts them,
   while pair 1's frames (`receive_capacity`) and the piece ring hold them; once the last is posted, its slot's receive
   is posted again (`stripe_listen`). The 256 slots so bound the descriptors in flight: credit flow control holds the
   peer's next descriptor until a slot is posted again (TN3205), and the peer's pieces until their receives are.
   This is the bridge's one header-driven receive, and it carries stripes alone (bridge-and-ipc.md#lanes).

**The guard** (`port_guard`, `guard_raise`, `guard_lower`). On a port whose device holds a stripe flow's queue pairs,
every change of the device's verbs objects on the port's queue (a flow's `flow_work` jobs: `pair_up`, `pair_receive`,
`pair_send` and the first receives posted; its `down_pair` in `flow_release`; every lanes job; every stripes job) runs
with the guard raised: `guard` made odd, and the job waits until each started progress thread has stored that value in
`guarded[side]` (an address wait on each word, the thread waking it as it stores). A thread checks `guard` at the top
of its loop, before each pass, and again before it parks; while it is odd the thread makes no pass (no post and no
poll on any queue pair of the device), waits on the bell, and tells the port's `epoch` as it moves, so a `port_settle`
inside the guarded job returns. The guard is lowered (`guard` even) and the bell rung when the job's verbs calls
return. A stripe flow's own jobs always raise it. With no stripe flow on the port, nothing is guarded and HEAD's
jobs run as before.

**Words and failures, as the clients see them:**

| event | writer's `word` | peer's `word` | ring |
|---|---|---|---|
| the stripe's last SEND completed / its last piece landed | `value` | `value` | — |
| refused by the event loop or the send thread | `UINT64_MAX` | untouched | `failed + 1`, `code` |
| the stripe flow failed or closed with it in flight (`stripes_abandon`) | `UINT64_MAX` | untouched (no acknowledgement crosses back, as at 8fc27cbe) | `failed + 1`, `code` `ECONNRESET` |
| the writer's slot retired meanwhile | untouched (its rows stay held while `flight` is nonzero: `clients_turn`) | as above | — |

**Trust.** As at 8fc27cbe: a neighbour's descriptor names any offset of this node's window and any word in it, and
nothing at the peer checks who wrote what (any node on a cable may already write into the window: bridge-and-ipc.md
#exec).

**Deviations from 8fc27cbe**, each for HEAD's interfaces or for the October 1 diagnosis:
1. A request names its destination node (`node`), never a port (8fc27cbe's `link`): clients name nodes on the
   resident bridge (#clients). Every cable toward the node takes requests.
2. The service is a flow of the port, made on demand with STRIPES over the link's control socket and served by the
   port's two progress threads, which park; 8fc27cbe ran a `mesh.stripes` thread per link that never parked, paired
   its own two queue pairs over a pairing connection of key 0 for the bridge's life, and made, connected and
   destroyed them on that thread.
3. One owner of every teardown (the port's queue), the guard, and the flush before destroying (above).
4. A descriptor slot is posted again only once its stripe's receives are all posted; 8fc27cbe copied each landed
   descriptor into a 256-entry ring and reposted its slot at once, so more than 256 descriptors landed ahead of their
   pieces' receives overran the ring.
5. The piece is the lesser of the two ends' (each its device's `max_msg_sz`, at most 4 MiB), told in STRIPES.
6. Refusals and losses are values (the writer's word `UINT64_MAX`, the ring's `failed` and `code`); 8fc27cbe had none.
7. A ring is a lease of a client slot, deregistered at its retirement and reclaimed with its rows; 8fc27cbe's
   outlived a process that died without `mesh_detach`.

**Integration check** (the main session, both bridges at MESH_VERSION 120; never measured on the pair, at 8fc27cbe or
since):
- On each node a client leases the same exact pages (`mesh_section_at`) and a word; the M5's client fills its range
  with a pattern (page index and word index, as `rdma/mesh-stream.c` writes), writes 1 GiB as 256 stripes of 4 MiB
  toward the Mini, and waits on its word; the Mini's client (no program of the key) waits on its word and checks every
  word of the range. Read the bridges' "stripes up ... pieces of N bytes" and, at the end, "stripes closed (N
  completions flushed)".
- Reproduction target: the raw stream's 9.383 GB/s one way (design/transport-one-gib-2026-09-10.md, `mesh-stream
  raw`); the stripe path's rate is the bridge's overhead on it.
- The guard: E2B pair calls (sessions made and released) while one node writes stripes continuously; a session's
  teardown with the stripe flow up must crash nothing (the October 17 `tbt_post_recv + 492` signature) and fail no
  flow's first SENDs with `IBV_WC_LOC_PROT_ERR`.
- The teardown: stop the Mini's bridge with a stripe flow up, then the M5's, apart; the kernel log names each
  extent's deregistration once and no panic follows (design/RDMA-KERNEL-RECOVERY.md "The stripe writes of October 1").

## Streams

The one-way 1 GiB reading of a link, raw verbs against the bridge's own path, every word checked
(design/transport-one-gib-2026-09-10.md: raw 9.383 GB/s, the bridge's page path 9.330-9.347), came from two standalone
tools, mesh 554a3793^ `rdma/mesh-verbs-stream.c` (a process opening its own verbs context) and cea7fce7^
`rdma/mesh-page-stream.c` (a client of that day's bridge), deleted on 2026-09-12 as unlicensed leftovers. A process
other than the bridge opens no verbs now, so the raw stream runs inside the bridge, at a client's request, and the
bridge's path is a client program; one tool asks for both (`rdma/mesh-stream.c`, `make -C rdma mesh-stream`, at
`rdma/.build/mesh-stream`).

**Raw** (`mesh-stream raw NODE BYTES` → `mesh-dataflow.h mesh_stream` → a request of kind `MESH_REQUEST_STREAM` (4) to
`[::1]:18519`, `node` the receiver, `bytes` the stream's; the bridge's answer and a JSON line come once the stream ends):

| step | sender's bridge (the requester's) | receiver's bridge |
|---|---|---|
| ask | `stream_asked`: the first cable toward `node` (`EHOSTUNREACH` none; `EBUSY` a stream on the port; `EINVAL` under a block); its lease (`bytes` in whole blocks from the window, `ENOMEM`), filled with the pattern word w of page p = `p << 32 ^ w ^ 0x6d65736870616765` (`streams_work`, a global queue) | — |
| make | `pair_up(1)` (the port's queue, guarded) → STREAM `count` = OFFER (its QP and PSN, `code` = the bytes) | `stream_heard`: its lease cleared (`EBUSY`, `ENOMEM` → STREAM FAILED), `pair_up(1)`, `pair_receive` (the sender's QP), `pair_send`, its first `window` receives posted, the threads started, then STREAM POSTED (its QP) |
| connect | POSTED → `pair_receive`, `pair_send` | — |
| run | the send thread (`stream_pass`): SEND completions taken, then the next pages posted as one chain while fewer than `window` are in flight | the receive thread (`stream_pass`): receive completions taken (a whole page each, in posting order, else `EPROTO` or a misordered count), the 1st, 10 %, 90 % and last completion timed (`ledger_ns`, `CLOCK_UPTIME_RAW`), the next receives posted as one chain |
| end | the last completion: `finished`; waits for the result | the last landing: `finished` → every word checked (`streams_work`), then STREAM RESULT (`struct stream_result`) |
| answer | RESULT → the requester answered 0 and one JSON line: `bytes`, `page_bytes`, `pages`, `window`, the sender's posts and submit-to-last, the receiver's completed pages, mismatched words, correct pages, misordered completions, first-to-last time and bytes, `interior_s`, `interior_bytes` and `interior_GBps` (10 % to 90 % of the pages, the 2026-09-10 rate) | — |
| close | `up = 0`, settle, guarded flush and destroy, the lease returned, "stream ... closed (N completions flushed)" | the same |

A page is one block (16 KiB on Apple silicon), so a GiB is 65,536 pages, and `window` is the lesser of 1,023
(mesh-verbs-stream.c's), the queue pair's frames over a page's (4,095 / 4) and the completion queue's room. A stream
waits on its events alone: a verbs or completion error, the peer's FAILED or the link lost ends it as a value at both
ends and answers the requester with the code. The stream's
queue pair is a restored form's: its jobs always raise the port's guard, and while it exists every other verbs change
on the port is guarded too (#stripes).

Deviations from mesh-verbs-stream.c: the bridge, not a process of its own, opens the queue pair, on the port's queue,
over the link's control socket rather than its own TCP exchange; the progress threads post its chains (their pass
also looks at the port's other flows, idle in a reading); one queue pair a direction and one stream a port at a time.

**The bridge's path** (`mesh-stream receive NODE BYTES` on the receiver, then `mesh-stream send NODE BYTES` on the
sender): a program named `mesh-stream-<bytes>` of one transfer of `bytes` (one invocation, `bytes / block` chunks) on
queue 0, bound, prepared and started as any client's (`mesh_transfer_bind`, `mesh_transfers_prepare`,
`mesh_host_inputs` at the receiver: a landing word a block); the sender fills its lease with the pattern, publishes the
one cell (`mesh_host_publish`) and detaches (its bridge posts the cell's SENDs, takes their completions and releases
its flow); the receiver follows its landing words in order until every block landed or one was cancelled (the 1st,
10 %, 90 % and last timed), checks every word and prints the same fields as the raw line, then detaches. This is mesh-page-stream.c's reading on HEAD's flows: the bridge posts the
cell's 65,536 SENDs and the ring's receives within the queue pair's frames (1,023 pages in flight), each landing a word
store.

**Reproduction target** (design/transport-one-gib-2026-09-10.md, M5 to the Mini): raw 9.383 GB/s interior, the
bridge's path within 0.5 % of it (9.338). The mb ledger's "link best" of 9.21-9.25 GB/s (metal-microbench
docs/measurement.md) is below the raw reading; the raw line, read the same day, is the denominator of a crossing's
efficiency.

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
the port's `max_msg_sz`; one link-local GID a port. Its polls never sleep: a completion queue that stayed empty more
than 8 polls yields the core (`sched_yield`), so N bridges' threads on one host's cores let the processes with work
run. The first process to need the fabric makes and sizes the segment; another that opens it first looks at its size,
yielding, until the maker's `ftruncate`, which follows its `shm_open`. Deviations from 800fe6bc^: the segment's name from the bridge's
configuration (`loop_fabric`), not an environment variable; device names `rdma_loop0` to `rdma_loop63`; a queue pair's
ring taken by compare-exchange of its `used` word, so numbers are reused only once free (800fe6bc^ took the next of
1,024 modulo, clobbering a live ring past 1,024 makings); the dropped SEND; the flush; rings of 8 MiB (1 MiB).

**A loop bridge** (`make -C rdma mesh-flow-loop`, `rdma/.build/loop/mesh-flow-loop CONFIGURATION`): mesh-flow.c built
with `MESH_LOOP_FABRIC` and loopverbs in place of `-lrdma`. Its configuration (the canonical property list, written by
`mesh-loop up`, not by `bridge_configuration.py`) adds `loop`: `region` (its shared memory object, in place of
`/mesh0`), `node` (its number, in place of the host UUID's fold), `service` (its clients' requests' loopback port, in
place of 18519), `fabric`, and `ports`, each a loopverbs `device` and its cable's two loopback ports (`listen`, this
end's; `peer`, the other end's). `configuration_read` refuses one without them (`loop_read`). `mesh-loop up` states the
deployed bridges' `keepalive` (5 us) and a `hello` of 1 s in each. Its ports are those devices alone, in that order
(`ports_find`); a port's turn opens its device and the cable's lower port dials the higher's listener, which accepts it
as the port's control socket (`port_opened`, the event loop's `TAG_LOOP`), in place of a link's hello and the lower
node's dial (a dial refused because the higher bridge does not listen yet is dialed again at the next `hello`); the
HELLO, records, map, lanes, flows, relays, stripes and streams are the deployed bridge's. Its name is its
configuration's (`host_watch` reads no host name), and it posts `MESH_STARTED` followed by its region's name.

**A loop client** (`make -C rdma host`): `rdma/.build/host/libmesh.dylib` (libmesh with `MESH_LOOP_FABRIC`:
`mesh_loop_bind(region, service)` names the bridge a process attaches to and asks, before its first attach or
request) and `rdma/.build/host/libnccl-mesh.dylib` (nccl-mesh.c with `MESH_HOST_EXECUTOR` and nccl-mesh-host.c in
place of nccl-mesh-metal.m and mesh-metal.m: every function of nccl-mesh-metal.h run on the CPU in the walk's order,
transcribed from the kernels, its waits polls of the words the bridge stores, ended by a landing or a cancel (recorded
in the violation words, which end the program's later waits), so the Metal path's walk runs with no GPU kernel; the
Metal 4 form's own steps are refused as the classic form refuses them). Deviation from 800fe6bc^: HEAD's Metal half (the violation words, combine3,
counted all-to-all, split-KV partials, retain and contents) transcribed; the arithmetic is nccl-mesh.c's own
(`nccl_mesh_combine`, `nccl_mesh_premultiply`, `nccl_mesh_truncdiv`, hidden, where `MESH_HOST_EXECUTOR` is defined).

**The driver** (`rdma/.build/host/mesh-loop`, rdma/mesh-loop.c; metal-microbench b9dbc43^ tools/nccl_demo.py `loop`'s
part that needs no Python):
Every wait of the driver is on an event (a map's serial, a bridge's start, a process's exit, a landing word, a pipe),
none bounded by a clock:
- `mesh-loop up FAMILY N WINDOW_MIB DIRECTORY`: a line, ring, star, complete map or a torus of two rows, of N (2 to 8)
  nodes, every cable two loopverbs devices and two free loopback ports; each node's configuration written, its bridge
  started (posix_spawn, its log in DIRECTORY), and the call returns once every bridge's map holds N nodes and its own
  cables are paired (`mesh_nodes_health`, `mesh_observe`), each bridge awaited on its map's serial (`mesh_map_serial`,
  `mesh_map_wait`), or on its start (`mesh_bridge_wait`) where it does not hold its region yet.
- `mesh-loop relay DIRECTORY BYTES`: every ordered pair, one after another, a program of one transfer of BYTES from
  the one to the other (the routed pairs through the relays between, on lanes), the sender detaching once it published,
  every word checked at the receiver.
- `mesh-loop stripes DIRECTORY BYTES`: every cable each way, a stripe of BYTES (#stripes) to the same exact pages
  (`mesh_section_at`) near the far window's top, the far reader with no program of its own, every word checked.
- `mesh-loop nccl DIRECTORY COUNT host|walk`: N ranks, all-reduce, all-gather, reduce-scatter, broadcast and reduce
  from every root, and a send/receive around the ring, COUNT float32 each, every value exact, on the host path or the
  Metal path's walk on the host executor.
- `mesh-loop destroy DIRECTORY COUNT RUNS host|walk`: RUNS times, N ranks, an all-reduce then a last group (a send and
  receive around the ring) of COUNT float32 each, every value exact, every rank's `ncclCommDestroy` right after its last
  group; then every bridge holding no flow (`mesh_busy`), and the bridges' logs read: flows ended in order, relays
  released by both their ends, flows failed, and each exclusive flow's flush ("flushed N of M": N equal to M).
- `mesh-loop kill DIRECTORY COUNT host|walk [bridge]`: N ranks, an all-reduce, then the last rank waits while the
  others begin a second and is killed (SIGKILL): each other rank's call fails as a value, the time from the kill to its
  return read; with `bridge`, every rank begins the second and the last rank's node's bridge is stopped (SIGTERM, as
  launchd stops one) in its place: every rank's call fails as a value, that node's rank by its watch of its bridge's
  exit (`mesh_watch`), the others by the FAREWELL. Run last: the stopped bridge stays stopped.
- `mesh-loop down DIRECTORY`: SIGTERM to every bridge and each exit awaited (`EVFILT_PROC`), the fabric unlinked.

The ranks of `nccl`, `destroy` and `kill` meet at a unique id the driver makes before it forks them
(`ncclGetUniqueId`: its root's listener opened in the driver and inherited, so every rank's connect meets it
listening).

**Readings at mesh f6432540** (metal-microbench output_data/dead-history-20261017/scratch-delete/w1/run2, a line of 4 at
256 MiB windows): the NCCL matrix at 64 elements 48 of 48 on the host path and 48 of 48 on the walk, and at 65,536 on
both; relays 12 of 12; stripes 6 of 6; `destroy` 30 runs each on the host path and the walk at 4,096, 65,536 and
1,048,576 elements, 0 landings cancelled, 360 flows ended in order and 120 relays released each 30 runs, no flow
failed, every flush exact, every bridge idle after; `kill` 18 of 18 survivor calls failed as values in 1.1-2.5 ms;
`kill ... bridge` on the walk 4 of 4, the stopped bridge's own rank in 0.9 ms. A ring of 4 whose loop binary was
replaced between calls stopped all four bridges between calls (`EVFILT_VNODE`).

**Reproduction targets** (800fe6bc^, its readings with the GPU walk on the host executor): the NCCL matrix on an 8-node
line 2,346 of 2,346; line 8, shuffled ring 8, star 8 and torus 2x4 104 of 104 each; ring 5 with routes 46 of 46. The
matrix itself (every datatype, op and root, metal-microbench b9dbc43^ tools/nccl_demo.py) is the consumer's driver's to
restore on these targets; `mesh-loop nccl` is its float32 core. The integration check: `mesh-loop up line 8 512 D`,
then `relay D 4194304` (56 pairs, the farthest 7 hops), `stripes D 4194304`, `nccl D 65536 host` and `nccl D 65536
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
   - Every QP is UC (`mesh-verbs.h pair_up`). Nothing in the code establishes what a UC SEND completion
     waits for on this provider.
6. **When RECORD is sent.**
   - The #protocol table's RECORD row omits the periodic send.
   - `health_turn` sends this node's record on every port at the configuration's `health` interval and on urgent
     health changes (recovery.md#diagnosis describes this).

Outside bridge-and-ipc.md: heterogeneity.md §1 (line 28) says a link's beta is "8e9 over the widest paired port's
bandwidth". `mesh-dataflow.c mesh_paired` and `mesh-plan.h` use the slowest port.
