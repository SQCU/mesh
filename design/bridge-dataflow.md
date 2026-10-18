# The bridge's data flow

Where each byte and word of the mesh bridge goes: who writes it, who waits on it, and what happens when it fails. The
reasons for the design are in [bridge-and-ipc.md](bridge-and-ipc.md) (the bridge), [recovery.md](recovery.md) (loss
and recovery) and [heterogeneity.md](heterogeneity.md) (link costs and shares); this page repeats none of them.

Conventions. The page describes the code at mesh `2594a087` (`MESH_VERSION` 123). A file name with no directory is in
`rdma/`. A citation is `file function`. "Word" means an 8-byte atomic in the shared region. "Block" is
`hdr.block * hdr.pgsz`. The bridge always starts with `block` = 1 (`mesh-flow.c main`: `mesh_layout(&geometry,pg,1,...)`),
so a block is one 16 KiB page on Apple silicon, and a row is one block. A "frame" is 4,096 bytes of a request (the
path MTU): a request of `n` bytes is `ceil(n / 4096)` frames.

In one paragraph: the bridge registers its window once on every port's device; at a link's first pairing it makes the
link's two completion queues and its queue pairs (pair 0 the keepalive's, pairs 1 and up the link's leases) and never
destroys them, making one again in place (reset) where its holder lets go of it or fails, and every one at the link's
next pairing after a loss; it tears no verbs object down, and the kernel reclaims them all at its exit; a program's
members agree which leases each of their transfers rides, and through which relays, at the bridge of the first member;
the bridge compiles each started program's SEND chains and receive rings onto those leases, and each port's two progress
threads spin over the link's two completion queues, posting what the clients release and storing what lands.

## 1. Processes, threads and memory

### Processes and threads

| process / thread | started by | does | waits on |
|---|---|---|---|
| `rdma/mesh-flow`, the bridge (one per node) | the launchd agent `io.mesh.bridge` (`bin/mesh-user.sh` writes it with `ProgramArguments` = mesh-flow and the canonical configuration plist, `KeepAlive`, `RunAtLoad`, stderr to `~/Library/Logs/io.mesh.bridge.log`); the keeper loads it at boot | everything below | — |
| the event loop (the bridge's main thread) | `mesh-flow.c main` | ports, control sockets, PAIRS, the map, client slots and programs' starts and ends, request channels and the `meet` and `relay` services, health, the stop | `kevent64` with no timeout: `EVFILT_USER` (`wake`), the listener, the HELLO socket, the kernel's network events, control and request sockets, `EVFILT_PROC` on client and `exec` child pids, `SIGINT`/`SIGTERM`/`SIGHUP` (`EVFILT_SIGNAL`), `EVFILT_VNODE` on its binary, the power sources' and thermal pressure's notify descriptors, and the configuration's `hello` and `health` intervals (`EVFILT_TIMER`) |
| `control_watch` thread | `main` | turns a change of `hdr.control` into a `wake()` of the event loop; returns at `stop` | `os_sync_wait_on_address(&M->control)` |
| port p's receive thread `mesh.rdma.receive` (side 0) and send thread `mesh.rdma.send` (side 1), QoS user-interactive | `port_threads`, at the link's first pairing (`pairs_heard`); kept for the bridge's life | side 0: `receive_pass`, the link's receive completions (landing words stored, receives posted, keepalive slots posted again); side 1: `send_pass`, the link's SEND completions (gates retired), keepalive frames, the leases' SEND streams, closing rings' drained bits | none while the link is up and no guard stands: a pass after a pass (`port_progress`); while the link is down or the guard stands, `port->bell` |
| port p's work queue `mesh.port` (serial dispatch queue) | `ports_found` | the port's device: opened and the window registered on it once, its port queried at every turn (`device_up`, `device_gid`); only while the port has no control socket (`port_turn`); never released (the kernel reclaims it at the bridge's exit) | its jobs |
| finder queue `mesh.ports` | `main` | the node's device list (`ports_find`), the host name at each change configd notifies (`host_watch`, `host_changed`), the kernel's memory-pressure changes (a dispatch source: `pressured`, `wake`) | — |
| a client process (a rank, anything linked to libnccl-mesh or libmesh) | its user | attaches to the region, leases rows, binds transfers, prepares, agrees its leases, starts programs, publishes and consumes | its sockets during the agreement, its destinations' `status[d].prepared`, its own landing words |
| a client's bridge watcher (one thread a process) | `mesh-dataflow.c mesh_watch`, at the first `mesh_open` of a bridge | at the bridge's exit: `bridge_pid = 0`, each slot of this process its words cancelled and each destination's status `ESRCH` with its start woken, the map's and health's serials moved by 2, the map, health and control words woken | `kevent64`: `EVFILT_PROC` `NOTE_EXIT` on the bridge's pid |
| the client's GPU work | the client's command buffers | publishes into SEND cells; spins on landing words | `mesh.metal mesh_remote_wait`, nccl-mesh-metal.m `mesh_step_wait`, ended by a landing or a cancel |

### The shared region `/mesh0`

The bridge creates the region at start: `shm_unlink`, `shm_open(MESH_NAME, O_CREAT|O_RDWR, 0666)`, `ftruncate`, `mmap`
(`mesh-flow.c main`). It lays the region out with `mesh.h mesh_layout`. Every section is page-aligned and the arena is
block-aligned:

```
offset 0      struct hdr
planes_off    3 row bitmaps (MESH_ROW_OWN, MESH_ROW_HOT, MESH_FREE), one bit per row
arena_off     arena occupancy bitmap, one bit per arena page
page_off      struct mesh_page_entry[rows]   mapping (slot index << 32 | arena page), address (region offset)
buffer_off    struct mesh_buffer[rows]       owner (client owner word), closed, pages (lease extent, head row)
link_off      struct mesh_link_info[8]       the port table
lease_off     uint64_t[8][MESH_LEASES = 16]  each port's lease words
client_off    struct mesh_client[64]         client slots
delivery_off  struct mesh_delivery[rows]     device_input (region offset of a landing word), device_stride
data_off      ARENA = the window: arena pages, registered whole on every port's device
```

At start, before the region is announced, the bridge stores every page entry's mapping `MESH_ABSENT`, every lease word
`MESH_LEASE_DOWN`, maps the arena's alias (`wire_map`), puts its own record in the map (`record_changed`), stores
`bridge_pid`, then `magic` last. Every client or reader that maps the region checks `magic`, `version` (`MESH_VERSION`)
and `length`, and that `bridge_pid` is alive (`mesh-dataflow.c mesh_open`), and watches that bridge's exit
(`mesh_watch`). A region of another version is refused with `EINVAL`; a dead bridge gives `ESRCH`. A client that waits
for a bridge to start blocks in `mesh_bridge_wait` on the notify(3) name the bridge posts once its sockets listen and
its control watch runs (`MESH_STARTED`, a loop bridge's followed by its region's name).

**Header words** (`mesh.h struct hdr`):

| word | written by | read / waited by | meaning |
|---|---|---|---|
| `magic`, `version`, `pgsz`, `block`, `rows`, `node`, `links`, `clients`, offsets, `length` | bridge at start (`main`); `magic` last | clients at attach | geometry; `node` is a 31-bit fold of `gethostuuid` (a loop bridge's its configuration's) |
| `bridge_pid` | bridge (start; 0 at the stop, before its exit); a client's watcher (0 at the bridge's exit) | every client wait (`mesh_gone`) | the bridge's process |
| `serial` | `mesh_slot_take` (clients and the bridge's relay slots) | — | slot generation counter (24 bits used) |
| `control` | clients (`mesh-call.c mesh_transfers_start`, `mesh-dataflow.c mesh_retire`, `mesh_detach`); bridge (at its stop) | the bridge's `control_watch`; a client's watcher wakes it at the bridge's exit | "look at the client slots now" |
| `map`, `map_off`, `map_nodes` | bridge (`map_put`, `map_grow`); a client's watcher moves `map` by 2 at the bridge's exit | `mesh-dataflow.c mesh_records_read` (seqlock, an odd value waited on with no bound), `mesh_map_serial`, `mesh_map_wait` | the map: odd while the bridge writes it |
| `health` | bridge (`health_put`); a client's watcher (by 2, at the exit) | `mesh_nodes_health` | the seqlock of a record change that changes only health, sequence and port counts |

**Lease words** (`mesh_leases(m, p)[l]`, `MESH_LEASES` = 16 a port). Word l of port p names lease l of the port's
link, which is the link's queue pair l + 1:

| value | written by | meaning |
|---|---|---|
| `MESH_LEASE_DOWN` (`UINT64_MAX`) | bridge: at start for every port (`main`), at the port's finding (`ports_found`), at its link's loss (`port_lost`); at pairing, for every l with l + 1 ≥ the link's pair count (`pairs_heard`) | no queue pair l + 1: the link is down, or the link has fewer pairs |
| 0 | bridge at pairing (`pairs_heard`), at a holder's release (`path_release`: compare-exchange holder → 0), for a holder whose slot is gone and whose pair no path holds (`clients_turn`); a client freeing a claim the agreement did not keep (`mesh_transfers_agree`); the bridge freeing a refused relay's claims (`relay_build`) | free |
| a client slot's owner word | a client's agreement (compare-exchange 0 → its owner, every free lease of every paired port, `mesh_transfers_agree`); the bridge for a relay slot (`mirror`: from 0, or from a not-started program of the same key) | held: the program of that owner rides queue pair l + 1 |

`program_take` starts a program only on leases whose words hold its own owner (else `ENETRESET`). The agreement reports
a port only where its link table entry is `MESH_PAIRED` with `leases` > 0.

**The port table** (`struct mesh_link_info`, 8 entries, bridge-written): `peer` (the paired node, 0 none) and `phase`
(`MESH_STOPPED`, `MESH_PAIRING`, `MESH_PAIRED`) and `code` (the last stop's) by `link_set`; `bandwidth`, `peer_port`
(the index the peer gave the cable) and `pairings` (one more at each HELLO heard on the control socket) by `port_heard`;
`max_pairs` by `port_opened`; `pairs` (the link's pair count) and `leases` (that count less one) by `pairs_heard`, 0 by
`port_lost`. A change of peer or phase is a new record (`record_changed`), and so is the link's pairing, so the flooded
record carries its lease count. `phase` is `MESH_PAIRED` from the peer's HELLO on the control socket; the leases exist
from its PAIRS (§2).

**A client slot** (`mesh.h struct mesh_client`, 64 slots):

| field | written by | meaning |
|---|---|---|
| `owner` | the client takes the slot with a compare-exchange 0 → owner (`mesh_slot_take`); `mesh_retire` (abort) sets 0, `mesh_detach` sets `owner \| MESH_RETIRING \| MESH_ORDERLY`; the bridge vacates an orderly slot once its program ended (`pairs_turn`, `clients_turn`) | `(generation << 38) \| (slot << 32) \| pid`; 0 vacant |
| `configured` | the client, compare-exchange 0 → owner (`mesh_transfers_start`); 0 at retire | the program is handed to the bridge |
| `key` | the client (`mesh_slot_take`: FNV-1a of the execution's name, `mesh_key`) | the execution; the same on every member and on its relays' slots |
| `depth`, `cyclic`, `targets`, `target_count` | the client (`mesh_transfers_prepare`, `mesh_transfers_cyclic`) | ring depth; whether the program is cyclic (M30); the program's publications sorted by row |
| `tx[d]` (64), `struct mesh_tx` | the client (`mesh_transfer_bind`, `mesh_transfers_prepare`, `mesh-collective.c mesh_host_inputs`, `mesh_transfers_agree`); the bridge for its relay slots (`mirror`) | the destination `node`; the lists at `orders` with `length[4 queues x 2 directions]`, `room`, `lists`; `cells` (region offset of the destination's first SEND cell); `cancel` (region offset of the cancellation record); `slots`, `invocations`, `count`, `once`; `port` (this node's port index + 1 toward the destination or its route's first hop) and `lease[4]` (each queue's lease index + 1, 0 none), both from the agreement |
| `status[d]`, `struct mesh_port_info` | the bridge (`status_prepared`, `status_failed`); the client's watcher at the bridge's exit; zeroed at take | `phase` (`MESH_PAIRED` once started, `MESH_STOPPED` once failed or its peer's end read), `code`, `domain`, `failed` (the node named), `prepared` (the owner once the bridge started or failed it: the client's start waits on it) |

**Inside a client's leases** (arena pages owned through rows). A client writes its rows and its slot; the bridge writes
the statuses, the port table, the lease words it owns as above and the map, and also these words inside a client's
leases:

- **SEND cells** (`mesh.h struct mesh_send`, 256 B, 128-aligned). One column per slot of each SEND transfer: one cell
  for each active invocation, plus one terminal cell that no producer releases. All of them sit in one lease from
  `mesh_transfers_prepare`, followed by the program's `struct mesh_target`s. The client writes `request.wr_id = 1` (the
  producer's argument) at prepare and `ready` (argument + cycle) at each publication (`mesh-collective.c
  mesh_host_publish`; `mesh.metal mesh_publish`). The bridge writes `span` and `request` (chunk 0's request), `successor`
  (the stream's next cell: the next invocation's, a cyclic program's last cell followed by its first, a one-pass
  program's by its column's terminal cell), `chunks`, `queue` and `pair` (0) (`mesh-flow.c sends_configure`). In a relay
  slot the bridge's receive thread writes `ready` too (§3).
- **Landing words and the cancellation record** (`mesh-collective.c mesh_host_inputs`). One `uint64_t` for each
  received chunk of each invocation, per destination, each destination starting on a 16-word boundary. The words of
  destination d form `cancel->ranges[d]` (`mesh.h struct mesh_cancellation`); every `tx[d].cancel` names that one record.
  `mesh_host_inputs` writes each received row's `mesh_delivery` (the word of invocation 0 and the per-invocation stride
  `8*count*chunks`). The receive thread stores 1 + cycle into a word as its chunk lands (§3); the bridge or the client's
  watcher stores `~0` to cancel it (§4).
- **Rings and payload**: the client's own sections (`mesh-call.c mesh_section_create`). The NIC writes received payload
  into them by DMA; senders write payload into them (`memcpy` on the host, Metal kernels on the GPU).
- **Transfer lists** (`struct mesh_transfer`, 48 B), leased per destination (`mesh-call.c mesh_lists_grow`).

**The window and the bridge's two mappings.** The arena is mapped twice in the bridge. `M` maps the whole region.
`wire.data` is an alias of the arena alone, at a 4 GiB-aligned base (`mesh-verbs.h wire_map`), because `tbt_post_recv`
loads only the low 32 bits of an SGE address (RDMA-KERNEL-RECOVERY.md). Every SGE address is in `wire.data`
(`wire_span`). A client maps the region once, at its own address; its leases are region offsets. A GPU sees the
client's SEND-cell lease and its landing words as `MTLBuffer`s created with `newBufferWithBytesNoCopy` over the client's
mapping (`mesh-metal.m mesh_metal_transport_create`).

**The bridge's own arena leases**: the map's records (`map_grow`, leased again twice as large as the mesh grows), one
page a port for its keepalive frames (`ports_found`: sixteen 64-byte SEND frames at the page's start, sixteen receive
slots after them), and the leases of its relay slots (`mirror`, through `mesh_section_create` on a slot whose owner pid
is the bridge's own).

### Verbs objects and their owners

| object | per | made | destroyed | owner |
|---|---|---|---|---|
| context, PD | port (its device) | `mesh-verbs.h device_up` on the port's queue (`port_turn`), each where it is not yet: once for the process's life | never: the kernel reclaims them at the bridge's exit; a verbs call the open device refuses, a registration aside (it turns again), stops the bridge (`device_lost`), and launchd starts it again on a fresh context | `port.device` |
| MRs | device: the window in 1 GiB extents (`MESH_REGION`), `IBV_ACCESS_LOCAL_WRITE` only | `device_up`, before any queue pair exists, checked against `max_mr` (`EFBIG`); a refused extent keeps the ones registered; `device_up` stops between extents at `stop` | never: the kernel's reclaim at the exit | `device.regions[]` |
| the link's two CQs: `sent` (SEND completions) and `received` (receive completions) | link | `link_make` at the peer's HELLO on the control socket (`port_heard`), each where it is not yet (mesh-verbs.h `link_cqs`): `min(n * frame_capacity, max_cqe)` entries each, n the link's pairs; at each later pairing what they hold from the pairing before taken and dropped | never: kept through the link's loss (`link_quiet`); the kernel's reclaim at the exit | `port.sent`, `port.received` |
| pair 0, the keepalive's UC QP | link | `link_make` (`pair_new`: created and INIT at the device context's first pairing, made again in place at each later one); joined at the peer's PAIRS (`pair_join`: RTR, RTS, its 16 receive slots posted) | never: let go of at the link's loss with no verbs call (`link_quiet`), its queue pair, round and capacities kept | `port.pair[0]` |
| pairs 1 to n − 1, the leases' UC QPs | link | `link_make`, as pair 0; made again in place (`pair_new`, mesh-verbs.h `pair_reset`: ERR, RESET, INIT, its number kept, its `round` one more) where its holder lets go (`path_release`), where the peer's newer making is heard and no program holds it, or where it failed with none holding it (`lease_remake`) | never: as pair 0 | `port.pair[k]`; `pair.path` while a program holds lease k − 1 |

n is `min(max_pairs, LINK_PAIRS = 17)` at this end, and the link uses the fewer of its two ends' counts (`pairs_heard`):
a TB5 device's `max_qp` 11 is `max_pairs` 10 (one held back: RDMA-FIRST.md), so a TB5 link is 10 queue pairs, its
keepalive's and 9 leases, as a loopverbs link is. Every queue pair is `IBV_QPT_UC` (`pair_make`) on the link's two CQs,
its send and receive capacities `frame_capacity` = `min(max_qp_wr, QD = 4095)` (less than `max_cqe`), one SGE, path MTU
4096, a GRH with hop limit 1 and the port's GID (`device_gid` takes the first non-empty entry), the peer's LID and GID
from its HELLO on the control socket, a random 24-bit PSN at each making (a making again in place keeps the queue pair's
number and draws a new PSN). Each pair's `share` of a CQ is its entries
over n; a pair's SEND gate is `min(sends, share)` frames and its receive ring's capacity `min(receives, share)` frames,
so neither CQ overflows. Only `IBV_WR_SEND` and `post_recv` are posted, each with one SGE, every SEND signaled. There is
no RDMA WRITE or READ: TB5 offers SEND and RECEIVE alone (TN3205).

A work request's `wr_id` is `round (16 bits) << 48 | pair index (8 bits) << 40 | low (40 bits)` (`wr_id`): the low bits
are a receive record's index in its ring, a lease SEND's gate position (frames posted, cumulative), a keepalive receive
slot, or a keepalive SEND's position. A completion whose pair index is past the link's pairs, or whose round is not its
pair's current one, is skipped (`receive_pass`, `send_pass`): it is an earlier making's, among them the flushed
completions of a pair's move to the error state as it was made again in place.

**The port's guard** (`guard_raise`, `guard_lower`). Every change of a device's queue pairs or CQs (made, connected,
made again in place, a CQ's entries of the pairing before dropped; none is destroyed) and every change of what holds a
pair (a path attached or released, the link's pairs let go of at its loss) runs on the event loop with the port's
`guard` raised: `guard` is made odd and `bell` rung, and `guard_raise` returns once each running progress thread has
told, in `quiet[side]`, that it holds at that guard value (or returned at `stop`: `UINT64_MAX`). The threads pass again
once `guard_lower` makes it even and rings `bell`. So no queue pair of a device is posted on or polled while another of
it changes, and the threads never read a `pair.path` or a ring mid-change.

### Diagram

```
 node A                                                                               node B
 ┌─ client process ────────────────┐   ┌─ mesh-flow (io.mesh.bridge) ──────────────────────┐       ┌─ mesh-flow ─┐
 │ host: memcpy payload -> ring    │   │ event loop: kqueue (sockets, NOTE_EXIT, signals,  │       │   (same)    │
 │       mesh_host_publish ->      │   │   vnode, EVFILT_USER <- control_watch, threads)   │       │             │
 │       cell.ready = 1+cycle      │   │ port p: queue (device open/register/query)        │       │             │
 │ GPU:  mesh_publish -> ready     │   │   send thread:  cell.ready -> post SEND on lease ─┼─UC────┼─> RECV ─┐   │
 │       mesh_remote_wait <- word  │   │   recv thread:  CQ -> store word, post next recv  │ TB5   │ recv thread │
 │ agree: meet/relay channels      │   │   pair 0: KEEPALIVE frames                        │       │ stores word │
 │ start: configured -> control    │   │ control socket TCP [fe80::%enN]:18519 ────────────┼─TCP───┼─ control    │
 └──────────┬──────────────────────┘   └──────────┬────────────────────────────────────────┘       └──────┬──────┘
            │ mmap                                 │ mmap (M) + wire alias of the arena (4 GiB-aligned)     │
            ▼                                      ▼                                                        ▼
 ┌──────────────────────────────── /mesh0 on node A ────────────────────────────────┐     /mesh0 on node B
 │ hdr: magic version node bridge_pid serial control map map_off map_nodes health    │
 │ planes | arena bits | page entries | buffers | port table[8] | lease words[8][16] │
 │ client slots[64]: owner configured key depth cyclic targets tx[64] status[64]     │
 │   tx: node cells cancel orders length[8] port lease[4]                            │
 │ delivery[rows] (row -> landing word, stride)                                      │
 │ ARENA (window, registered whole on every device):                                 │
 │   client leases: rings, SEND cells+targets, landing words+cancel record, lists    │
 │   bridge leases: map records, keepalive pages, relay slots' leases                │
 └───────────────────────────────────────────────────────────────────────────────────┘
```

## 2. The control plane

### Ports and peers

A port is one RDMA device of the node (`rdma_<interface>`), found in device-list order on the finder's queue
(`ports_find`, at the start and at an interface attach or link-on that no port has) and added on the event loop
(`ports_found`): its keepalive page leased from the window, its work queue made, its lease words `MESH_LEASE_DOWN`, its
port table entry `MESH_STOPPED`, `wanted`. A port turns at events (`port_turn`, only while it has no control socket and
no job): its device opened where it is not and the window registered on it once, then its port read (`device_up`,
`device_gid`) on its queue, which wakes the loop (`done`). `port_opened` takes the result:

- a registration refused: logged with the extent; the port turns again at its next memory-pressure change (`pressured`)
  or link event;
- a device that failed with its context open (any error but `ENETDOWN`, `ECANCELED`): the bridge stops (`device_lost`,
  `MESH_STOP_DEVICE`), and launchd starts it again on a fresh device context;
- open: `max_pairs` in the port table, the port's LID, GID and bandwidth read, the entry `MESH_PAIRING`, and a HELLO
  sent to `ff02::1` on the port's interface (UDP, port 18519).

`hello_heard` takes a HELLO from another node on an open port with no control socket: the higher node answers with its
own HELLO, the lower dials the higher's listener at the address it heard from (non-blocking `connect`; its write event
completes it). The listener (`[::]:18519`) gives a link-local connection to the port of its scope id; a loopback
connection is a client's request (§2, Requests). Either end then makes the socket non-blocking and says HELLO on it
(`port_greet`). The first message on a control socket must be a HELLO with the same block size, from the peer the port
expects if any (else `EPROTO`, the link lost). `port_heard` then: `greeted`, `peer`, `peer_port`, the HELLO kept (its
LID and GID address the peer's queue pairs), the port table's `bandwidth`, `peer_port` and `pairings`, `link_make`
(below), the entry `MESH_PAIRED` (a new record), and every record of the map sent to the peer.

A port with no peer turns again at the configuration's `hello` interval (`HELLO_TIMER`), where it states one. A link
lost (`port_lost`) is `wanted` again, so a link that had paired says HELLO again at the port's next turn.

### PAIRS: a link's queue pairs, rounds, codes and `sent`

`link_make` makes the link's CQs where they are not yet and its n pairs (`pair_new`: `pair_make` creates each and moves
it to INIT, its `round` one more, a new PSN, its gate `min(sends, share)`, its counts zero; for a pair an earlier
pairing made, kept through the loss with its round, `pair_new` resets it in place instead, `pair_reset`: ERR, RESET,
INIT, its number kept) under the guard, takes and drops what the CQs hold from the pairing before, and tells the peer
every pair (`pairs_tell(LINK_PAIRS)`): a PAIRS message, `count` = n, and an entry a pair (`struct pairs_entry`: `index`,
`qpn`, `psn`, `round`, `code`, `sent`).

`pairs_heard`, at the link's pairing (`port->count` 0): the link's count is the fewer of the two ends' n; each pair
below it takes the higher of the two ends' rounds (an end whose pairs were kept through a loss is rounds ahead of one
whose bridge started again, and a completion of a round before stays skipped) and is joined to the peer's of its index
(`pair_join`: RTR to the peer's `qpn` and `psn`, RTS, the keepalive pair's 16 receive slots posted) under the guard; the
lease words below the count free (0), the rest `MESH_LEASE_DOWN`; the port table's `leases`; the port's two progress
threads started (once for the bridge's life); `up` = 1 and `bell` rung; the region's link record its `pairs` (n) and
`leases`, and the node's record flooded at once (`record_changed`); logged "node N paired: n queue pairs, n − 1 leases,
`share` frames a pair". A count under 2 is the link's loss (ENOSPC); a join refused stops the bridge (`device_lost`), as
does a pair's making or making again refused (`link_make`, `lease_remake`, `path_release`).

After the pairing, PAIRS tells one pair made again (`pairs_tell(k, code, sent)`), from `path_release` (its holder let
go), `lease_remake` (the peer's newer making answered, or a failed pair with no holder) and `link_make`. For each entry:

- an older round than this end's pair: ignored;
- the same round as this end's pair: joined (`pair_join`), under the guard;
- a newer round, on pair 0: `EPROTO`, the link lost;
- a newer round where a program here holds the pair, with `code` ≠ 0: the peer's holder failed with it, and this
  program fails as a value (`program_fail(code, domain 4, the peer)`);
- a newer round where a program here holds the pair, with `code` 0: the peer's holder ended in order; `peer_gone` set
  and `peer_sent` = `sent`, the SENDs that holder completed on the pair's making before; once this pair's ring has
  landed that many (`pairs_turn`), the program here lets go of the lease (`peer_released`, §3);
- a newer round where nothing here holds the pair: this end's made again in place to the peer's round and told
  (`lease_remake(k, 0)`), and the peer joins it as it hears that.

A pair is joined only where the peer's last told making is this end's `round` (`pair_join`), so the two ends of a lease
always run the same making, and each end makes its pair again (in place) only at its own release or failure, or to
follow the peer's; until then a pair made again waits in INIT and takes nothing.

### Messages on a control socket

A message is a `struct message` header (`magic` = `0x4d595048 + MESH_VERSION`, `kind`, `node`, `count`, `key`, `code`,
`block`, `port`, `from`, `to`, `bytes`, `lid`, `gid`) and `bytes` of payload (at most 64 MiB, else `EPROTO`). `port_say`
queues a message behind what the socket has not taken and writes as far as it takes (`port_flush`; the rest at its
`EVFILT_WRITE`); a send that fails is the link's loss. Before the HELLO only HELLO is said.

| kind | payload | sender | receiver |
|---|---|---|---|
| `HELLO` | — (`node`, `block`, `port`, `lid`, `gid`) | `port_greet`; on UDP, `port_opened` and `hello_heard` | `port_heard` (first message), `hello_heard` (UDP) |
| `PAIRS` | `struct pairs_entry[]`, `count` the sender's n | `pairs_tell` | `pairs_heard` |
| `FAREWELL` | — (`code`: the stop's reason, a signal's number, `MESH_STOP_BINARY` or `MESH_STOP_DEVICE`) | the stop (`main`) | `port_heard`: `parted`, the link lost (`ECONNRESET`), logged "parted" with the reason |
| `RECORD` | `struct mesh_record` | `record_changed` (its own, on every greeted port), `record_heard` (flooded, or sent back), `port_heard` (every other node's record of the map, to a new peer) | `record_heard` |
| `CHANNEL_OPEN`, `_CONFIRM`, `_REFUSE`, `_DATA`, `_WINDOW`, `_SHUT`, `_CLOSE` | the service's name (OPEN), bytes (DATA) | `channel_say` | `channel_heard` |

Any other kind is `EPROTO` and loses the link. A link's pair 0 carries `KEEPALIVE` frames (a `struct message` with
`kind` `KEEPALIVE`, 64-byte frames), which the receive thread counts and posts again; nothing reads their contents.

### The map and records

Each node's record (`struct mesh_record`: node, name, sequence, window, memory, model, its ports' table entries, its
health) is built from the port table (`record_changed`, its sequence one more) at each change of a port's peer or
phase, a link's pairing (its `pairs` and `leases` set), a rename, an urgent health change, and the configuration's `health` interval; it is put in the map and sent on
every greeted port. `record_heard` (RFC 2328 s13): this node's own record from an earlier start of its bridge, newer
than its own, moves its sequence past it; another node's record no newer than the newest known of it is dropped (the
newer one sent back where this end's is newer; a record the map pruned put back only where a paired path reaches it);
a newer one is put in the map and flooded on every other greeted port. `known` holds the newest record of up to 256
nodes, map or not.

`map_put` writes the map between two notifications of `hdr.map` (odd while it writes), only where the records kept
differ: the record put in, and every node no paired path from this node reaches left out (logged "node N unreachable"
with its last health). A record whose links, name and facts equal the held one's changes only health, sequence and
port counts, and is written under `hdr.health` instead (`health_put`), so the map serial moves only where the mesh
changes. The records live in arena pages the bridge leases (`map_grow`). A client reads them whole by the seqlock
(`mesh_records_read`) and builds the map: two nodes joined by one link where each one's record holds a paired port to
the other, at the slowest such port's bandwidth, and the routing table (`mesh-dataflow.c mesh_whole_make`,
`mesh_route`). A node's bridge carries a pair of neighbours' programs over one cable, the lowest index the pair's lower
node gave one of its cables to the other (`cable_to`; the agreement's `agree_port_to`).

### Client slots: `clients_turn`

At each turn of the event loop (after any event; a client's `mesh_notify(control)` wakes it through `control_watch`),
`clients_turn` reads every slot but the bridge's own:

| slot's `owner` vs the program the bridge holds for it | action |
|---|---|
| `base \| MESH_RETIRING \| MESH_ORDERLY` and the program running | `ENDING`, every path `closing` (§3, ends) |
| `base \| MESH_RETIRING \| MESH_ORDERLY` and the program failed | the program freed, the slot vacated (compare-exchange → 0) |
| `base \| MESH_RETIRING` | nothing yet (the client is still retiring it) |
| anything else (0: an abort, or a new owner) | `program_fail(ECANCELED, domain 4, this node)`, then freed |
| `MESH_ORDERLY` set and no program | the slot vacated |
| a new owner | its process watched (`EVFILT_PROC NOTE_EXIT`; a pid already gone retires the slot at once) |
| `configured == owner`, not retiring, no program, not yet taken | `program_take` (§3) |

Then every lease word held by an owner no slot holds, whose pair no path holds, is freed (an agreement cut short), and,
where no program of a gone owner remains, the rows of retired clients are released (`mesh_retired_release`). A watched
process's exit retires each slot it holds (`mesh_retire`, an abort). The bridge's own slots (its relay slots) are
skipped.

### Requests (channels)

A client's request is a loopback connection to its bridge (`[::1]:18519`, a loop bridge's `service`) sending a `struct
mesh_request` (`mesh-dataflow.c bridge_ask`), answered with an int32 code:

- `MESH_REQUEST_OPEN` (`mesh_request(node, service, bytes)`): a channel (RFC 4254 s5) to `service` at `node`, keyed by
  (origin node, sequence) on every hop, opened toward the next hop of this node's route to `node` (`channel_hop`:
  `mesh_route`, then `cable_to`) with `CHANNEL_OPEN` and a window of 128 KiB; each hop forwards it, and the destination
  opens the service (`service_open`, after `admitted`: a request for `bytes` outside the window is refused `ENOMEM`
  where the node's available memory is less) and answers `CHANNEL_CONFIRM`, or `CHANNEL_REFUSE` with the code; the
  client is answered 0 at the confirm, and the connection then carries the channel's bytes both ways, each hop holding
  at most a window each way (`CHANNEL_DATA` of at most 32 KiB, `CHANNEL_WINDOW` granting back), `CHANNEL_SHUT` for a
  half close and `CHANNEL_CLOSE` with a code for a failure. Services: a registered Unix socket; `nodeinfo` (loopback
  8100, else 8099); `exec` (where the configuration serves it: its first line run by the user's shell, the rest its
  standard input, its output then its int32 wait status; a channel that fails sends its process group SIGTERM); and the
  bridge's own `meet` and `relay`, below.
- `MESH_REQUEST_SERVE` (`mesh_serve`): a service name registered at a Unix socket path while the connection stays open.
- `MESH_REQUEST_LEDGER` (`mesh_ledger`): §5.

**`meet`** (`meet_heard`, `service_heard`): a member sends a `struct mesh_meet` (key, n, rank, bytes) and its report,
then half closes; the bridge holds each report until every rank 0..n − 1 of the key has sent one (a second report of a
rank ends the first one's channel, `ECONNRESET`), then answers each member with all of them (a uint32 n, n uint32 byte
counts, the reports in rank order) and ends its side. At most 256 ranks and 1 MiB a report.

**`relay`** (`relay_build`, `mirror`): the requester sends a `struct mesh_relay_request` (key, count, bytes) and its
entries, each a `struct mesh_relay_entry` (neighbours `a` and `b`, the leases toward each, slots, invocations, depth,
cyclic, the transfer counts each way queue by queue) followed by its transfers. For each entry the bridge takes a slot
of its own with the request's key (`mesh_slot_take`), mirrors the two directions into it (each far transfer received
from one neighbour and sent to the other over one section of `mesh_relay_slot` bytes), prepares it as its ends are
prepared, claims its leases toward each neighbour (free, or held by a program of the same key not yet started, else
`EBUSY`), and starts it (`program_take`, a relay program). It answers 0, or the refusal with every slot of the request
retired and its leases freed. The channel then stays open for the programs' life: a byte past the request is the
requester's word that it ends in order (`orderly`), and the channel's end drops the programs (`relay_drop`, §3).

A channel ends at its two ends' half closes (`channel_end(0)`), or with a code at a socket's error, a peer's
`CHANNEL_CLOSE`, the loss of a port it rides (`requests_lost`), or the stop (`requests_stop`: `ESHUTDOWN`). Its slot is
freed before its services let go (a meet's report dropped, a relay's programs dropped), so a relay's failure that loses
a link never ends the channel twice.

## 3. Programs

### Agreement: `mesh_transfers_agree`

After `mesh_transfers_prepare` (and `_cyclic`) and before `mesh_transfers_start`, each member of an execution calls
`mesh_transfers_agree(context, nodes, n, rank)` with the same `nodes` list (bridge node numbers; this node at `rank`).
Every member of several meets, whether or not its program binds a transfer; a lone member (`n` 1) that binds none
returns 0 at once.

1. **Claims.** For every port its bridge reports `MESH_PAIRED` with leases, the member claims every free lease word
   (compare-exchange 0 → its owner) and reports the port: its index, peer, `peer_port`, lease count and the mask of the
   leases it claimed. It reports each destination it binds: node, slots, invocations, depth, cyclic, its transfers each
   way queue by queue.
2. **Meet 1.** The report goes to the `meet` service of `nodes[0]`'s bridge under the program's key (`agree_meet`);
   every member reads back every report.
3. **The root's plan** (rank 0, `agree_plan`): for every pair of members that binds a transfer either way, its route on
   the root's bridge's map (`mesh_route`; fewer than 2 hops or more than 256 is `EHOSTUNREACH`); for each hop and each
   queue the pair uses, the lowest lease index that every member end of that hop claimed (`agree_port_to`: the cable
   both ends take) and that no other pair of this assignment holds on that link (`EBUSY` where none); each end's port
   and leases recorded; each relay node on the route given an entry (its two neighbours, the leases toward each, the
   transfers of the direction pair). Each relay node is asked once with all its entries over the `relay` service, and
   its answer read; the channels are kept in the root's context (`context->relays`). A list of more than one slot a
   queue is `EINVAL`.
4. **Meet 2** (the key xor `AGREE_SECOND`): the root reports the assignment (a code, then an entry a member end: rank,
   far node, port, the four lease indices + 1), the others an empty report; every member reads it.
5. Each member writes `tx[d].port` and `tx[d].lease[]` for its entries and frees every claim the assignment did not
   keep; on a refusal it frees them all, and the root closes its relay channels (each relay's programs then fail as an
   abort).

Every member returns the same refusal where the root's plan refused. The agreement waits on its sockets alone: before
each blocking read or write it stores the descriptor in `context->agreeing` and reads its own slot's owner (a retired
slot is `ECANCELED`), and `mesh_transfers_interrupt` (libnccl-mesh's `session_retire`: a revoke or a cancel) shuts that
descriptor down from another thread.

### Start: `mesh_transfers_start` and `program_take`

The client sets `configured` (compare-exchange 0 → owner), notifies `hdr.control` and waits on each destination's
`status[d].prepared` (§4, Waits). The bridge's `clients_turn` takes the program (`program_take`), within the event
loop's turn, and never waits on a peer:

1. A `struct program` (owner, relay or not, `RUNNING`, invocations, cyclic) and a path a destination with transfers
   (`port` = `tx[d].port − 1`, and `cancel` = `tx[d].cancel`, the cancellation record `words_cancel` reads whoever
   holds the slot later), with a ledger where the configuration keeps them.
2. Refused at once as a value, every destination's status `MESH_STOPPED` with the code, domain 1, naming this node:
   `EINVAL` (no node or this node; a transfer of more slots than the program's or past `MESH_QUEUES`; a chunk outside the
   window; a lease index 0 or past the link's pairs), `ENETDOWN` (its port not up), `ENETRESET` (a lease word not this
   owner's: the link paired again since the agreement), `ENOMEM`.
3. **Compile** (`sends_configure`, `receives_configure`): each queue's transfers become a ring r (one slot a queue).
   SENDs: every active cell of each SEND transfer gets its chain of chunk requests over its invocation's ring slot
   (chunk 0's request and SGE in the cell, the rest in `path->requests`, each `IBV_WR_SEND` of one block-bounded span in
   the device's registration, `wire_span`), and the cells are linked in invocation order and binding order within one
   into the ring's stream (`successor`). Receives: each ring's records (`struct prepared_receive`: request, SGE, the word
   its landing stores, frames, invocation, ring, chunk, binding, position) in invocation, binding and chunk order; each
   record's word is the delivery's landing word where it is the program's own (`receive_input`), the next hop's SEND
   cell where the program also publishes the row (a relay: `release`), else the record's own `sink`; each record's
   `behind` is the record its ring must have landed before it is posted (the ring's span a slot short of its depth;
   a cyclic program's first records behind its last of the cycle before).
4. **Attach** (`path_attach`, under the guard): each record's `wr_id` named by its lease pair's round and index, each
   ring's capacity `min(receives, share)`, `pair.path` and `pair.ring` set, `holding` one more, the ring's first records
   posted (`ring_advance`). A post refused fails the program.
5. Every destination's status `MESH_PAIRED`, code 0, `prepared` = owner, its waiter woken.

### One transfer through the two loops and the gate

1. The producer writes the payload into its ring slot and releases the cell: `ready` = 1 + cycle (`mesh_host_publish`,
   or the GPU's `mesh_publish`).
2. The send thread's pass (`send_pass`): SEND completions taken (each retiring its pair's gate to its position), a
   keepalive frame where due (below), then for each lease pair held by a path, connected and not failed, its stream
   (`stream_post`): while the stream's current cell has `ready` > the stream's cycle, its chain posted request by request,
   each signaled with `wr_id(round, k, posted)` while `posted − retired + frames ≤ gate` (`ibv_post_send` accepts past
   the queue's frames and corrupts later, so the gate is the poster's); a cell posted whole moves the stream to its
   `successor`, and wrapping to its first cell is the next cycle. A post refused is the pair's failure.
3. The peer's receive thread (`receive_pass`): a receive completion of its pair's current round names its record; the
   ring's `landed` one more, its outstanding frames less, the record's word stored `argument + cycle` (1 + cycle) with
   release order, a ledger stamp, the ring's next records posted while capacity holds them and each one's `behind` has
   landed (`ring_advance`), and the event loop woken where `landed` reaches the pair's `peer_sent`.
4. The consumer sees the word reach 1 + cycle (`mesh_host_arrived`, `mesh_remote_wait`, nccl-mesh.c `landed`).

TB5 holds a SEND until the peer's receive for it is posted (TN3205's credit), so the two ends' programs start apart: a
SEND released before the far program's receives exist waits in the hardware.

**The keepalive.** Where the configuration states `keepalive` and a program holds a lease of the link (`holding`), the
send thread posts a 64-byte KEEPALIVE frame on pair 0 once that long passed since the last one, with at most 16 in
flight; the peer's receive thread counts it and posts its slot again. A frame keeps the link's Thunderbolt from going
quiet under a program (the Mini wakes slowly after quiet: mesh 12f484f8); it detects nothing.

**Progress threads.** `port_progress` (side 0 receive, side 1 send) loops while `stop` is clear: with the link up and the
guard even, one pass after another, no wait; with the guard odd or the link down, it tells `quiet[side]` = the guard
value and, where neither has changed, waits on `bell` (no timeout). At `stop` it stores `quiet[side]` = `UINT64_MAX` and
returns. Each pass is bounded: a poll takes at most 32 completions, a stream's posts end where a cell is not released or
the gate is full, a ring's posts where its capacity or a `behind` stops them. A thread's error on a pair (an error
completion, a post refused, a CQ poll failing) is `pair_failed`: the pair's `failed` word set once and the event loop
woken; the event loop takes it (`pairs_turn`).

### Relays: forwarding and closing

A relay program is a bridge's own slot with two paths, one toward each neighbour, whose received rows are the rows it
sends from: the last chunk of a record that lands from one neighbour stores 1 + cycle into the SEND cell of the same
row and invocation toward the other (`receive_input`'s `release`), and the send thread posts that cell on the other
neighbour's lease. Each relay hop is a pair of leases the root's plan chose; the relay's data never reaches a client.
A relay program ends:

- **in order**, as its neighbours let go of their leases: where one neighbour's in-order release reaches this relay
  (`peer_released`), every other path of the relay is `closing`, so what landed from that side is sent on before the
  other side lets go in its turn (its own drained SENDs, then its release with its own `sent`); with no lease attached
  it has ended (`pairs_turn`: logged "ended in order", the program freed, its slot retired);
- **as a failure**, by a code told on one of its leases (its other leases then told the same code), a pair's error, a
  link's loss;
- **as an abort**, where its channel ends without the requester's byte past the request (`relay_drop(orderly 0)`:
  `program_fail(ECONNRESET, domain 4, this node)`, the slot retired). A channel that ends in order leaves its programs
  running to their end in order (`relay_drop(orderly 1)`).

The root's `mesh_detach` writes the byte `e` on each relay channel where its own slot ended in order (not retired before:
`ncclCommDestroy` of a session that neither failed nor was revoked), then closes it.

### Ends

| end | starts at | the bridge | the peer and the far end |
|---|---|---|---|
| **in order** | `mesh_detach` (`ncclCommDestroy` after its GPU work finished): `owner \| MESH_RETIRING \| MESH_ORDERLY` | `ENDING`, every path `closing`; the send thread sets a ring's `drained` bit once its stream has no released cell left to post and its pair's SENDs all completed (`posted == retired`), or the peer let go of the lease; with every ring drained, `path_release(code 0)`: each lease pair made again in place and told with code 0 and `sent` = its SEND completions on the making before, its lease word freed; with no path attached, "ended in order" and the slot vacated. The landing words are not cancelled; the client waits on none of this | each peer holding the lease: `peer_gone`; once its ring has landed `sent` (`pairs_turn`), `peer_released`: its status toward that destination `MESH_STOPPED`, `ECONNRESET`, domain 4, naming the far node (where its client still owns the slot), its words kept; the lease let go (`path_release(code 0, only that pair)`) and made again in place; a relay passes it on (above) |
| **peer's in-order end** | a PAIRS entry of a newer round with code 0 on a held lease | as the right column | — |
| **abort** | `mesh_retire` (a revoke, `ncclMeshCommCancel`, `ncclCommAbort`, the client's process exit seen by the bridge) | `program_fail(ECANCELED, domain 4, this node)` and freed: its words cancelled though its client gave up the slot first, so the revoking process's own waits end; each lease made again in place and told `ECANCELED` | each peer holding the lease fails as a value with that code (below) |
| **failure** | a pair's error (`pairs_turn`: a lease's, domain 2, this node); a code told on a held lease (domain 4, the peer); the link's loss (domain 4, the peer; `ECONNRESET` where it parted); a relay's channel cut (above) | `program_fail`: every path's leases let go under the guard, its words past their landed counts cancelled whether or not the client still owns the slot, each lease made again in place (while its link is up) and told with the code; destinations never attached their words cancelled (by the path's `cancel`); every destination's status the failure where the client still owns the slot | each peer holding one of those leases fails alike, its own other leases told the code: the failure runs along every member and relay the program's leases touch |
| **link loss** | `port_lost` | §4 | the peer's own loss of the same link (its socket's end, its FAREWELL) |

A lease pair that fails with no program on it is made again in place and told with the code (`lease_remake(code)`); the
peer's holder, if any, fails by it. The keepalive pair's failure, or a CQ's, is the link's loss; a verbs call refused
while a pair is made again in place or joined stops the bridge (`device_lost`).

## 4. Failure, cancellation and release

### The `~0` landing word

A landing word holds 0 (not landed), 1 + cycle (landed in that cycle), or `~0` (cancelled). Writers of `~0`:

- `ring_cancel` (in `path_release` with `cancel`, from every `program_fail`, whether or not the client still owns the
  slot: the program's rows stay its own until it is freed): for each
  ring, every record from its first not landed through the program's invocations less the ring's depth (a cyclic
  program's readers keep within its depth of the landings), every one not landed in a one-pass program; never a
  relay's `release` cell or a record's own `sink`. Landed words stay.
- `words_cancel` (from `program_fail`, a destination whose rings were never attached, by the cancellation record the
  path saved at the take): every word of its range that is 0, or, in a cyclic program, any word not already `~0`.
- `mesh-dataflow.c mesh_words_cancel` (the client's watcher at its bridge's exit): the same rule over each of the
  process's slots' destinations.

A cancelled word ends a wait on it: the host path's polls fail the session (`session_watch`, `run`), and a GPU wait
that reads `~0` records it in its violation words, which end its program's later waits (nccl-mesh-metal.m
`mesh_step_check`). A program failed after its client gave up its slot (an abort: `mesh_retire` vacates it first) has its
words cancelled all the same, so a revoke or a cancel ends its process's own waits, a GPU walk's among them; only its
statuses, which a new owner of the slot may hold, are left alone.

### Statuses

`status_failed(status, owner, code, domain, where)` stores `code`, `domain`, `failed` = `where` and `phase` =
`MESH_STOPPED`, then `prepared` = owner, and wakes the start. Domains: 1, the bridge refused the start (`program_take`);
2, a pair of this node failed (an error completion's status, or a post's errno); 4, an event of the link or the peer
(a link's loss naming the peer, a code told on a lease naming the peer, an abort naming this node, a peer's in-order end
naming the far node, a relay's cut naming this node). The client's watcher writes `ESRCH`, domain 4, naming its node.

### How each failure reaches the clients

| event | detected by | this node | the other members |
|---|---|---|---|
| link off, interface detached (M24) | `network_heard` | `port_lost(ENETDOWN)`: programs on the link fail, its pairs let go of with no verbs call; the port turns again on its kept device context, and pairs again once its port is active (`KEV_DL_LINK_ON`, `KEV_DL_IF_ATTACHED` turn it); whether a TB5 link whose peer lost power pairs again on that context is unread (RDMA-KERNEL-RECOVERY.md "October 18") | the peer's own link-off |
| a verbs call the device refuses (a pair's making, making again or joining; the open device's port read, query or PD) | `port_opened`, `port_heard`, `pairs_heard`, `lease_remake`, `path_release` | `device_lost`: the bridge stops (`MESH_STOP_DEVICE`, below), and launchd starts it again on a fresh device context | the FAREWELL ("a verbs call of its device was refused") |
| control socket's end, a send failing, a protocol error | `control_heard`, `port_flush`, `port_heard` | `port_lost(code)` (`ENETRESET` at the socket's end, `ECONNRESET` after a FAREWELL; a read's or send's errno; `EPROTO`) | the peer's socket end |
| peer's FAREWELL | `port_heard` | `port_lost(ECONNRESET)`, logged "parted" with the reason | — |
| an error completion or a refused post on a lease | the progress thread (`pair_failed`), then `pairs_turn` | the holder's `program_fail(status or errno, domain 2)`; with no holder, the lease made again in place with the code | the code on the lease's PAIRS fails the peer's holder |
| an error on pair 0, or a CQ poll failing | as above | `port_lost(code)` | the peer's socket end at the loss |
| a code on a PAIRS entry for a held lease | `pairs_heard` | `program_fail(code, domain 4, the peer)` | the code passed on along this program's other leases |
| the client's process exit | `EVFILT_PROC` | `mesh_retire`, then `clients_turn`: `program_fail(ECANCELED)` | by the code |
| a revoke, `ncclMeshCommCancel`, `ncclCommAbort` | the client (`session_retire`: `mesh_retire`, `mesh_transfers_interrupt`) | as above, its words cancelled; an agreement in progress ends (`ECANCELED`, or its descriptor shut) | by the code |
| the peer's program ends in order | PAIRS code 0 with `sent` | status `ECONNRESET` naming the far node once `sent` landed; the lease made again in place; an `ncclMeshCommAgree` waiting on that peer ends (`ncclRemoteError`) | — |
| a relay channel cut | `relay_drop` | the relay's programs `ECONNRESET` | by the code |
| the bridge's exit | the client's watcher | words cancelled, statuses `ESRCH`, waits woken | the bridge's FAREWELL, or its socket's end |

### `port_lost`: a link's loss

1. The control socket closed; logged ("closed toward node N: this bridge is stopping", "node N parted, its bridge
   stopping (reason)", "node N lost: code").
2. `up` = 0; every lease word `MESH_LEASE_DOWN`; the port table's `leases` and `pairs` 0.
3. Every program with a path on the port fails as a value (`ECONNRESET` where the peer parted, else the code; domain 4;
   naming the peer). Its paths on this port let go of their pairs without making them again (the link is down) and
   cancel their words, whether or not the client still owns the slot; its paths on other ports make theirs again in
   place and tell their peers the code.
4. `link_quiet` under the guard, with no verbs call: each pair let go of, its queue pair, round and capacities kept for
   its making again in place at the next pairing (`link_make`), the CQs kept, the link's count and holdings zero.
5. Every channel riding the port ends with the code (`requests_lost`).
6. The port `wanted`, its socket state cleared; the port table `MESH_PAIRING`, or `MESH_STOPPED` at the stop (a new
   record).

### The client's watch of its bridge

`mesh_watch` (once a bridge pid, at a process's first `mesh_open` of it) maps the region writable where it can and
registers `EVFILT_PROC NOTE_EXIT` on the bridge's pid on one watcher thread a process (`mesh_watch_run`). At the exit
(`mesh_watch_exit`): `bridge_pid` = 0; for each slot this process owns, each destination's words cancelled, its status
`ESRCH` (domain 4, naming the node) where not already stopped, `prepared` = the owner and its waiter woken; `map` and
`health` moved by 2 (their parity kept, so a reader that read the bridge alive and has not yet slept finds its word
changed); the map, health and control words woken. No wait of the process outlives its bridge: a start reads `ESRCH`,
a seqlock reader `ESRCH`, a GPU wait `~0`.

## 5. The ledger

Where the configuration's `ledger` is true (`ledger_kept`), each path of each program keeps a ledger (`struct ledger`):
two rings of `MESH_LEDGER_EVENTS` = 65,536 events, SEND events (`struct ledger_send`: the cell, when the send thread
first saw it released, when its last request was posted, the cycle; `ledger_seen`, `ledger_posted`) and landings
(`struct ledger_landing`: the record, when its completion was taken, the cycle; `ledger_landed`), and its cells' names
(`struct ledger_cell`: cell, bytes, ring, invocation, position, binding, chunks; sorted by cell for lookup). The send
thread writes the SEND events and the receive thread the landings, each publishing its count with release order; about
3.5 MiB a path, outside the window.

A client's `mesh_ledger(name, path)` (nccl.h `ncclMeshLedger`) sends `MESH_REQUEST_LEDGER` with the program's key and an
absolute path. `ledger_cut` appends, for each path of each running program of the key, a header line (the ledger's id,
the key, this node, the path's far node and port, its invocations and cyclic, the event counts and the cut) and the
events since that path's last cut: `{"s":[id, ring, invocation, position, binding, cycle, chunks, seen, seen, posted, 0,
bytes]}` and `{"r":[id, ring, invocation, position, binding, chunk, chunks, cycle, landed]}`; then the cut moves to the
counts. `ENOENT` where no program of the key keeps a ledger, `ENOTSUP` where the configuration keeps none. `ledger.py`
pairs the two nodes' events.

## 6. Configuration operands

**The bridge's operands.** The bridge has one operand: the canonical plist that `bin/mesh-user.sh` writes from the
consumer's JSON, TOML or YAML document through `bridge_configuration.py load`, to
`~/Library/Application Support/io.mesh.bridge/configuration.plist`. Nothing has a default. `configuration_read` reads
up to 64 KiB of it.

| key (document) | canonical | effect on the data flow |
|---|---|---|
| `purpose` | `purpose` | none (required, for people) |
| `window_gib` / `window_percent` (exactly one) | `window` (bytes; a percentage is resolved against the node's memory on the node that installs it) | the arena is `window / pgsz` pages = rows (`mesh_layout`); every lease, ring, cell, landing word, map record, keepalive page and relay lease comes from it; a window outside [1 page, 2^32 pages] is refused at start; every device registers it whole, and a device with fewer than `ceil(window / 1 GiB)` MRs refuses the port (`EFBIG`; the port turns again at its next memory-pressure change or link event) |
| `exec` (optional bool) | `exec` | `exec_served`: a channel to `exec` spawns `$SHELL -c` (`child_spawn`); otherwise `EACCES` |
| `ledger` (optional bool) | `ledger` (written only if true) | `ledger_kept`: §5 |
| `keepalive_us` (optional) | `keepalive` (ns) | the send thread's KEEPALIVE frame on pair 0 while a program holds a lease of the link; none where unstated |
| `hello_us` (optional) | `hello` (ns) | the event loop's `HELLO_TIMER`: each port with no peer, no socket and no refused registration says HELLO again; none where unstated |
| `health_us` (optional) | `health` (ns) | the event loop's health timer: this node's record flooded again (`health_turn`); none where unstated |
| (the document's stem) | `name` | logged at start only |

Changing the plist takes effect only when the bridge restarts. `mesh-user.sh` copies the binary again on a
configuration change, so the binary-change stop applies.

**Fixed in the source** (each changes the data flow):

| constant | value | where | what it changes |
|---|---|---|---|
| `MESH_LEASES`, `LINK_PAIRS` | 16, 17 | `mesh.h`, `mesh-flow.c` | lease words a port; most pairs a link makes |
| `KEEPALIVE_SLOTS`, `KEEPALIVE_FRAME` | 16, 64 B | `mesh-flow.c` | keepalive frames in flight and receive slots; their size |
| `MESH_PORTS` / `MESH_CLIENTS` / `MESH_DESTINATIONS` / `MESH_QUEUES` / `MESH_ORDERS` | 8 / 64 / 64 / 4 / 4096 | `mesh.h` | region tables; rings a destination |
| `QD`, `MESH_REGION`, `MESH_BANK` | 4095, 1 GiB, 4 GiB | `mesh-verbs.h` | queue pair depth cap, MR extent, wire alias alignment |
| `MESH_CHANNELS`, `MESH_SERVICES`, `MESH_WINDOW`, `MESH_PACKET`, `MESH_EXEC_BYTES` | 256, 32, 128 KiB, 32 KiB, 4096 | `mesh-flow.c` | channels, registered services, a channel hop's window and packet, an `exec` command line |
| `MESH_MEET_RANKS`, `MESH_MEET_BYTES` | 256, 1 MiB | `mesh-plan.h` | a meeting's ranks and a report's bytes |
| `MESH_KNOWN` | 256 | `mesh-flow.c` | nodes whose newest record is known |
| `MESH_LEDGER_EVENTS` | 65,536 | `mesh-flow.c` | ledger rings |

**Operands of the clients that shape the bridge's traffic:** `NCCL_BUFFSIZE` (environment, default `MESH_BUFFSIZE` 4
MiB; `nccl-mesh.c setting`) sets the large slot through `mesh_wire_of`.

## 7. Where to look

`mesh-flow.c` by function, in file order:

| part | functions |
|---|---|
| verbs (mesh-verbs.h) | `device_gid`, `wire_span`, `wire_map`, `say`, `device_up`, `link_cqs`, `pair_make`, `pair_connect`, `pair_reset`, `port_bandwidth` |
| identifiers and wakes | `wr_id`, `wr_index`, `wr_round`, `now_ns`, `bell_ring`, `wake` |
| health and diagnosis | `available_now`, `farewell_text`, `sysctl_text`, `power_read`, `health_sample`, `reset_read`, `health_init`, `health_text`, `known_at`, `record_newer` |
| ledger | `ledger_seen`, `ledger_posted`, `ledger_landed`, `ledger_cell_order`, `ledger_free`, `ledger_cut` |
| statuses | `owned`, `status_prepared`, `status_failed` |
| compiling a program | `transfer_row`, `chunk_span`, `receive_input`, `sends_configure`, `receives_configure` |
| the two loops | `ring_advance`, `pair_failed`, `receive_pass`, `stream_post`, `send_pass`, `port_progress`, `port_threads` |
| the guard and the link's pairs | `guard_raise`, `guard_lower`, `pairs_tell`, `pair_new`, `pair_join`, `device_lost`, `lease_remake`, `link_make`, `link_quiet`, `pairs_heard` |
| releases and failures | `ring_cancel`, `words_cancel`, `path_release`, `path_attach`, `program_free`, `program_fail`, `peer_released`, `program_take`, `pairs_turn` |
| the map and records | `map_grow`, `map_put`, `record_links_same`, `health_put`, `record_held`, `record_changed`, `memory_watch`, `health_turn`, `record_heard`, `link_set` |
| channels and services | `fd_watch`, `fd_reset`, `fd_answer`, `channel_udata`, `channel_say`, `channel_end`, `channel_new`, `channel_hop`, `end_drain`, `end_put`, `channel_pump`, `end_read`, `admitted`, `service_open`, `child_spawn`, `child_ended`, `child_heard`, `asked`, `asking_heard`, `asking_accepted`, `channel_heard`, `request_event`, `requests_lost`, `requests_init`, `requests_stop` |
| relays and meetings | `cable_to`, `program_owned`, `relay_drop`, `mirror`, `relay_build`, `meet_drop`, `meet_heard`, `service_heard`, `service_gone` |
| configuration | `loop_number`, `loop_text`, `loop_read` (loop bridges), `configuration_read` |
| host name | `host_name`, `host_changed`, `host_watch`, `host_taken` |
| ports and control sockets | `port_watch`, `port_flush`, `port_say`, `port_heard`, `port_lost`, `port_greet`, `port_at`, `port_turn`, `port_opened`, `hello_heard`, `control_heard`, `ports_find`, `ports_found`, `network_heard` |
| clients | `clients_turn`, `control_watch` |
| start and stop | `stopped`, `main` |

Elsewhere: attach and slots, `mesh-dataflow.c mesh_attach`, `mesh_slot_take`, `mesh_program_attach`, `mesh_retire`,
`mesh_detach`; leases of rows, `mesh-call.c mesh_section_create`, `mesh-dataflow.c mesh_rows_alloc`, `mesh_arena_alloc`,
`mesh_lease_return`; transfers, `mesh-call.c mesh_transfer_bind`, `mesh_lists_grow`, `mesh_transfers_prepare`,
`mesh_transfers_cyclic`; landing words, `mesh-collective.c mesh_host_inputs`, `mesh_host_publish`, `mesh_host_arrived`;
agreement, `mesh-call.c mesh_transfers_agree`, `agree_meet`, `agree_plan`, `mesh_transfers_interrupt`; start,
`mesh_transfers_start`; the map, `mesh-dataflow.c mesh_records_read`, `mesh_map`, `mesh_route`, `mesh_map_wait`; the
bridge's life, `mesh_bridge_wait`, `mesh_watch`; requests, `bridge_ask`, `mesh_request`, `mesh_serve`, `mesh_ledger`;
sessions, `nccl-mesh.c session_open`, `session_agree`, `session_watch`, `session_close`, `session_retire`,
`session_ensure`, `ncclCommRevoke`, `ncclMeshCommCancel`; the configuration, `bridge_configuration.py load`,
`bin/mesh-user.sh`.

## Waits and their ends

Every wait in mesh-flow.c, mesh-dataflow.c, mesh-call.c, nccl-mesh.c, nccl-mesh-metal.m, nccl-mesh-host.c and
mesh-loop.c, the event that ends it, and what guarantees that event: the platform (the kernel, the verbs driver,
dispatch: a call that returns, an event it delivers), a live peer whose death is itself an event (a control socket's
end, a FAREWELL, a link-off, a process's exit, each turned into a failure or a cancel), the process's own threads, or
the caller (its revoke, cancel, exit or launch). No wait is ended by a clock, and no thread of the bridge sleeps with a
timeout.

**The bridge (mesh-flow.c).**

| wait | ends at | guaranteed by |
|---|---|---|
| the event loop's `kevent64`, no timeout (`main`); not entered where `stop` is set | any event it registered: `EVFILT_USER` (`wake`: a port job's end, a progress thread's pair failure, a landing that reaches a peer's `sent`, a ring drained, the control watch, the host name read, the device list read, a memory-pressure change), the listener, the HELLO socket, the kernel's network events, control and request sockets, client and `exec` child exits, `SIGINT`/`SIGTERM`/`SIGHUP`, its binary's `EVFILT_VNODE`, the power and thermal notifications, the `hello` and `health` timers | the platform; a signal also interrupts the wait (`EINTR`), and the loop reads `stop` after every event |
| `control_watch`: an address wait on `hdr.control` | a client's `mesh_notify(control)` (start, retire, detach); at the stop, the bridge's own after `stop` is set, which ends the thread (it reads the word before `stop`, so a bump after that read finds `stop` set) | the platform's address wake |
| a progress thread with the link up and the guard even | none: it passes again at once | — |
| a progress thread waiting on `bell` (the link down, or the guard odd), no timeout | a ring: `guard_raise`, `guard_lower`, the link's pairing (`pairs_heard`), the stop (`main` rings every port's `bell`) | the event loop's own rings; a ring between the thread's read of `bell` and its wait is not slept past (the wait compares the value it read) |
| `guard_raise`: an address wait on each started thread's `quiet[side]` | each thread telling the guard's value at its next loop head (a running thread ends its pass first; a waiting one wakes at the guard's ring), or `UINT64_MAX` at its return | the port's progress threads: every pass is bounded (above); a port whose threads are not started, or are joined, has nothing to wait for |
| `pthread_join` of a port's two threads (the stop) | each thread returning at its loop head once `stop` is set and `bell` rung | the bridge's own threads |
| `pthread_join` of the control watch (the stop) | the watch waking at `mesh_notify(control)` with `stop` set | the bridge's own thread |
| the process's exit (`main` returns; no wait on a port's queue) | the kernel's reclaim of every verbs object the process made, and every thread's end | the kernel: a port's job inside a verbs call that sleeps uninterruptibly in the driver (opening a device reads the device list, `ibv_get_device_list`, whose end the driver alone guarantees: recovery.md Q14) holds the process at its exit, which no signal ends (RDMA-RULES.md); whether the reclaim meets the driver's termination deadlock on TB5 is unread (RDMA-KERNEL-RECOVERY.md "October 18") |
| a program ending in order (`ENDING`) until every ring drained | its SENDs' completions; its peer's in-order release of each lease (`peer_gone`); a code told on a lease (its failure); a pair's error; the link's loss; the stop | the device: a UC SEND completes once sent; TB5 holds a SEND until the peer's receive is posted, so the peer's program running, its end, its failure or its link's loss. A peer whose program never attached its receives (its start refused before attaching, or its process gone between the agreement and its start) tells nothing: the program then ends only at the link's loss or the stop (recovery.md, Risks) |
| a held lease whose peer let go in order, until its ring has landed `sent` | the landings; this program's own end (in order: `peer_gone` drains it at once; an abort); its failure; the link's loss; the stop | the peer's SENDs, each completed on its side before it told `sent`; a SEND the wire dropped (UC) leaves it a landing short until one of the others |
| a running program awaiting its peers | none in the bridge: it posts and stores as completions come | — |
| a meeting (the `meet` service) | every rank of the key reporting; its channel's end (the member's socket or a port on its route lost, a second report of the rank, the stop) | the members' agreement calls; else the caller's revoke or cancel at the member |
| a channel | none: every end is non-blocking under the event loop (a socket's read or write event, a child's exit and its output's end, the port's loss, the stop) | the platform; the far bridge alive, its death the link's loss |
| a control socket's sends | none: a send the socket does not take is queued and written at its next `EVFILT_WRITE`; FAREWELL at the stop goes as far as the socket takes it | — |

**The stop.** A signal's handler (`stopped`: `SIGINT`, `SIGTERM`, `SIGHUP`) sets `stop_reason` and `stop` and does
nothing else; the event loop's `EVFILT_SIGNAL` sets them too, and its `EVFILT_VNODE` on the bridge's binary (its
contents written, or the file replaced, renamed or removed) sets `stop` with `MESH_STOP_BINARY`; a verbs call a port's
device refuses sets it with `MESH_STOP_DEVICE` (`device_lost`), and launchd starts the bridge again on a fresh device
context. Every loop reads `stop` every pass: the event loop before each `kevent64` and after each event, each progress
thread at its loop head and before it waits on `bell`, `device_up` between extents, `control_watch` after each wake.
After the event loop, `main`:

1. logs "stopping (reason)";
2. for each port, rings `bell` and joins its two threads;
3. ends every channel with `ESHUTDOWN` (`requests_stop`; a relay channel whose requester said it ends in order leaves
   its programs to the ports' loss below, the others fail them with `ECONNRESET`, their leases made again in place and
   their peers told the code while the links are up), and closes the asking and service sockets;
4. for each port, says FAREWELL with the reason and loses the link (`port_lost(ESHUTDOWN)`: every program on it fails
   as a value, its words cancelled; the link's pairs let go of with no verbs call, `link_quiet`);
5. frees every program; stores `bridge_pid` = 0; notifies `hdr.control` and joins the control watch;
6. logs "stopped: N queue pairs on P ports left to the kernel's reclaim at its exit" and returns 0, with no verbs call
   and no wait on a port's queue (a registration under way reads `stop` between extents); the `atexit` handler unlinks
   the region, and the kernel reclaims every verbs object, the mappings with them, at the exit
   (RDMA-KERNEL-RECOVERY.md "October 18").

The peers take the FAREWELL (or the socket's end) as the link's loss, and each client's watcher takes the exit.
`mesh-loop sigterm` sends SIGTERM to every loop bridge in each of eight states and reads each bridge's exit status, its
last log line, its region and the fabric's queue pairs still held by a live process (its own children reaped first):
`idle` (paired, nothing running); `pairing` (bridges started again and stopped once each region is open, before their
links pair); `host` and `walk` (N ranks each in a loop of all-reduces on the host path or the walk, stopped once each
finished one); `relay` (every ordered pair's relayed program in a loop, stopped once one landed); `linkloss` (the last
bridge stopped and the others at once, as its loss reaches them); `client` (the last rank killed in a call and every
bridge stopped at once); `peer` (the last bridge stopped and exited, then the others). At mesh 2594a087, a line of 4
three times and a torus of 6 once: 32 of 32 readings with every bridge exited 0 and its last line "stopped: ... devices
closed", no region left and 0 of the fabric's queue pairs held; from the SIGTERM to the last exit 0.9-3.0 ms on the line
and 1.4-5.8 ms on the torus (metal-microbench output_data/dead-history-20261017/scratch-rewrite/all.txt). At mesh
a68717f9, a line of 4 and a torus of 6 once each: 16 of 16 readings with every bridge exited 0, its last line "stopped:
... left to the kernel's reclaim at its exit" and no region left; the fabric's queue pairs held 0 in every state (the line's
`pairing` read 40 until the reading reaped the bridges it starts itself, whose unreaped exits still answered `kill(pid,
0)`; 0 in 3 of 3 after); 1.1-2.5 ms on the line and 1.4-7.7 ms on the torus
(metal-microbench output_data/mesh-noteardown-20261018).

**A client (mesh-dataflow.c, mesh-call.c).**

| wait | ends at | guaranteed by |
|---|---|---|
| `mesh_transfers_agree`: the `meet` and `relay` channels' connect, sends and reads (`whole_io`, `agree_meet`) | the meeting's answer (every rank reported); the root's assignment (meet 2); a relay's answer (`relay_build` answers within the relay bridge's turn); the channel's end (a port on its route lost, a bridge on it stopping, a second report of the rank); `mesh_transfers_interrupt` shutting the descriptor down; the slot retired, read before each blocking call | the other members' agreements, each of which ends likewise; a member that never reports holds every member's agreement until their callers' revoke or cancel |
| `mesh_transfers_start`: an address wait on each destination's `status[d].prepared` | the bridge's answer (`program_take` started or refused it, within its event loop's turn: it waits on no peer); the slot's retire (`ECANCELED`: `mesh_retire` wakes it); the bridge's exit (`ESRCH`: the watcher) | the bridge's turn at `mesh_notify(control)`; its death, the watcher |
| `mesh_records_read`, `mesh_nodes_health`: an odd seqlock value waited on (`mesh_whole_wait`) | the bridge's write ending (the word made even and notified); the bridge's exit (the watcher moves the word by 2, keeping its parity, and wakes it: the reader then reads the bridge gone, `ESRCH`) | the bridge, which holds the word odd only inside a write; its death, the watcher, in a process that can open the region for writing |
| `mesh_map_wait`: an address wait on `hdr.map` until it moves past `*serial` and is even | the bridge's next map write; its exit (`ESRCH`) | a change of the mesh, which the caller waits for; the exit as above |
| `mesh_bridge_wait`: a read of the `MESH_STARTED` notify descriptor, the region tried after each | a bridge's start (it posts `MESH_STARTED` once it listens), or at once where one runs | whoever starts the bridge: launchd's `KeepAlive`, or the loop driver; a bridge that exits before it holds its region (its configuration refused) leaves it waiting, its log saying why |
| the watcher thread (`mesh_watch_run`): `kevent64` on the watched bridges' exits, for the process's life | each watched bridge's exit | the platform; nothing waits on the thread |
| `bridge_ask` (`mesh_request`, `mesh_ledger`, `mesh_serve`): a loopback connect, then the answer read | the bridge's answer (at once for a local service, a ledger cut or a registration; at the far bridge's `CHANNEL_CONFIRM` or `CHANNEL_REFUSE` for a far one, which `service_open` answers at once), or the connection's end (refused where no bridge listens: `ESRCH`; closed at the channel's end, the bridge's stop or exit) | the platform; the far bridge alive, its death the link's loss and the channel's end |
| `mesh_calls_start`: a spin on each worker's `frame` | the worker thread running | the platform (`pthread_create` succeeded) |
| a call worker (`mesh_call_progress`) | `mesh_calls_destroy` (`running` 0) and every submitted function's completion (`mesh_call_complete`, `mesh_call_fail`) | the caller and its functions |

**libnccl-mesh (nccl-mesh.c, nccl-mesh-metal.m, nccl-mesh-host.c).**

| wait | ends at | guaranteed by |
|---|---|---|
| `comm->opening` (`pthread_cond_wait` on `opened`: ncclCommFinalize, ncclCommDestroy, a group's `session_ensure`) | the opening thread's open ending, done or failed | the open's own waits (the agreement, the start, `session_agree`), which end as their rows say |
| `session_lock` | the group of any communicator that holds it while it waits on its peers ending | that group's waits; `ncclMeshCommCancel` and `ncclCommRevoke` take only the communicator's own `guard`, so they reach a group waiting there at once |
| the agreement and the start (`session_open`, with `session_lock` released and `opening` the session) | as in the client's table | as there; a revoke or cancel retires the slot and shuts the agreement's descriptor (`session_retire`) |
| `session_agree`, `ncclMeshCommAgree`: each channel's position 0 (or `end`) polled, yielding | the peer's landing; its cancel (`~0`: a link on the way lost, the peer's program failed and its code told, this bridge's exit through the watcher); a revoke, read each turn; for `ncclMeshCommAgree`, the peer's destination status `MESH_STOPPED` with its position not landed (`peer_stopped`: its program ended in order or failed while it was held) | the live peer, its death or its end an event; or the caller |
| `run` (the host path): each channel's landing words polled | every position consumed and sent; a cancel (`~0`); the session's failure (`failed`, set by a cancel, a revoke or a cancel call) | as the agreement |
| the walk's GPU waits (`mesh_step_wait`, a spin in the kernel) and the host executor's (`step_wait`, yielding after 256 reads) | the word reaching 1 + the cycle; `~0`; the program's violation word [2] set (an earlier wait's cancel, which halts the program) | as the agreement. A revoke or cancel vacates the slot before the bridge fails the program, and the bridge cancels the program's words all the same (`program_fail`, by its paths' own records and saved cancellation records), so a revoked or cancelled session's GPU waits end there |
| `drain` (`metal_wait` on `finished`: `waitUntilSignaledValue` with no timeout; the host executor's polls, yielding) | every encoded program finished | each program's waits ending (the row above), then the platform's completion of its command buffer |
| the bootstrap (`ncclCommInitRank`): the root's `accept` of each other rank, then the reports read; a rank's `connect` to the root, then its read | each rank's connect and report; a connected rank's death is its socket's end (`ECONNRESET`) | the caller's launch of every rank (as NCCL's own bootstrap: a rank never started leaves the root's accept waiting); the connect: the platform (taken by the root's listener backlog, which `ncclGetUniqueId` opens before the ranks start where it is called in their parent, or refused at once where none listens) |

**The loop driver (mesh-loop.c).**

| wait | ends at | guaranteed by |
|---|---|---|
| `up`: each bridge's map serial (`mesh_map_wait`) or its start (`mesh_bridge_wait`) until every map holds N nodes and its cables are paired | each pairing's record, each start | the bridges it started; one that exits before its region leaves it waiting on its start |
| `relay`: a pipe's byte, the children's `waitpid`, a receiver's landing words polled (yielding) | the byte or the writer's exit (end of file); each child's exit; a landing or `~0` | the children's own waits |
| `nccl`, `destroy`, `kill`, `hop`, `share`, `revoke`: each rank's byte on its pipe, `waitpid` | each rank's result or exit | the ranks' library waits (above) |
| `revoke`: the revoking thread's 100 ms sleep before its revoke, then up to 10 s of 100 ms sleeps for its call's return; the never-calling rank's read of a pipe | the sleeps' ends (the one clock in the readings: it places the revoke inside the wait and names a wait the revoke does not end `hung`); the pipe's end once every other rank exited | the platform |
| `destroy`'s `bridges_idle`: each bridge's lease words and client slots read, yielding, until every lease is free or down and every slot but the bridge's own vacant | every program ended in order | each program's end in order (the bridge's table) |
| `kill`: `EVFILT_PROC` on the victim's bridge; the victim rank's `pause` | the bridge's exit (`SIGTERM`, or its binary replaced by rename: either stops it at once); the victim's `SIGKILL` | the platform; the driver sends both |
| `sigterm`: the workers' bytes on a pipe, then `EVFILT_PROC NOTE_EXITSTATUS` on each bridge after `SIGTERM`, then `waitpid` of the workers | each worker's first result; each bridge's exit; each worker's exit (its calls fail as values once its bridge stops) | the platform; the bridges' stop (above) |
| `down`: `EVFILT_PROC` on each bridge after `SIGTERM` | each exit | the platform |

## The loopback fabric

The operator: "our tensor parallelism code should be fast both on days when the 5 computers are a ring and also on days
when the 5 computers are a star" (metal-microbench docs/principles.md section 4), and "i cannot and will not precommit to
a specific ordering or routing!" (2026-10-03). Before the Studios arrive, the way to run the bridges' relays and
routes, the agreement on 3 to 8 nodes and the library's trees is N bridges on one host: `rdma/loopverbs.c`, the host
executor `rdma/nccl-mesh-host.c` and the `mesh-flow-loop` and `host` targets, transcribed from mesh 800fe6bc^'s loop
fabric onto the resident bridge. Their source's readings were the NCCL matrix on an 8-node line 2,346 of 2,346 and line
8, shuffled ring 8, star 8 and torus 2x4 104 of 104 each, and it found the 4-node ring+tree deadlock (c65445c) and the
cycle-0 defect (c5a4674), which "nothing at two nodes can show". The deployed `mesh-flow` is the same source without
`MESH_LOOP_FABRIC`.

**loopverbs** (`rdma/loopverbs.c`): the ibverbs calls the bridge makes, over a shared segment (its name the loop
configuration's `fabric`): a byte ring a queue pair, its number the ring's, held by the process whose queue pair holds
it (its `used` word that process's pid) and free again once that process has exited, as the kernel reclaims a process's
verbs objects at its exit (the bridge destroys none); each making and each move to `IBV_QPS_RESET` moves the ring's
generation; a ring is taken by compare-exchange of its `used` word from a holder that is gone, so numbers are reused
only once free. A SEND is copied into its destination's ring, its record's length word carrying the ring generation its
sender connected to, and completes as it is delivered (one to a queue pair gone, made again or reset since its sender
connected is dropped and completes, as UC drops it); the owner's polls drain its ring into its receives in the order
posted, dropping a record of an older generation (delivered before the pair's reset), a SEND waiting for its receive as
TB5's credit flow control holds it, `IBV_WC_LOC_LEN_ERR` for a SEND longer than its receive; a move to `IBV_QPS_ERR`
flushes the posted receives and SENDs (`IBV_WC_WR_FLUSH_ERR`); a move to `IBV_QPS_RESET` empties the pair's queues and
disconnects it until its next RTR, which records the peer ring's generation again; a device reports `max_qp` 11 (TB5's:
10 queue pairs usable, so a loop link is 10 pairs, 9 leases), `max_qp_wr` 4,095, `max_cqe` 65,535 and `max_mr` 1,024; a
SEND at most half a ring (rings of 8 MiB), the port's `max_msg_sz`; one link-local GID a port; devices `rdma_loop0` to
`rdma_loop63`. Its polls never sleep: a completion queue that stayed empty more than 8 polls yields the core
(`sched_yield`). The first process to need the fabric makes and sizes the segment; another that opens it first looks at
its size, yielding, until the maker's `ftruncate`, which follows its `shm_open`.

**A loop bridge** (`make -C rdma mesh-flow-loop`, `rdma/.build/loop/mesh-flow-loop CONFIGURATION`): mesh-flow.c built
with `MESH_LOOP_FABRIC` and loopverbs in place of `-lrdma`. Its configuration (the canonical property list, written by
`mesh-loop up`, not by `bridge_configuration.py`) adds `loop`: `region` (its shared memory object, in place of
`/mesh0`), `node` (its number, in place of the host UUID's fold), `service` (its clients' requests' loopback port, in
place of 18519), `fabric`, and `ports`, each a loopverbs `device` and its cable's two loopback ports (`listen`, this
end's; `peer`, the other end's). `configuration_read` refuses one without them (`loop_read`). `mesh-loop up` states a
`keepalive` of 5 us and a `hello` of 1 s in each. Its ports are those devices alone, in that order (`ports_find`); a
port's turn opens its device and the cable's lower port dials the higher's listener, which accepts it as the port's
control socket (`port_opened`, the event loop's `TAG_LOOP`), in place of a link's HELLO and the lower node's dial (a dial
refused because the higher bridge does not listen yet is dialed again at the next `hello`); the HELLO on the socket,
PAIRS, records, the map, leases, programs, relays, meetings and channels are the deployed bridge's. Its name is its
configuration's (`host_watch` reads no host name), and it posts `MESH_STARTED` followed by its region's name.

**A loop client** (`make -C rdma host`): `rdma/.build/host/libmesh.dylib` (libmesh with `MESH_LOOP_FABRIC`:
`mesh_loop_bind(region, service)` names the bridge a process attaches to and asks, before its first attach or request)
and `rdma/.build/host/libnccl-mesh.dylib` (nccl-mesh.c with `MESH_HOST_EXECUTOR` and nccl-mesh-host.c in place of
nccl-mesh-metal.m and mesh-metal.m: every function of nccl-mesh-metal.h run on the CPU in the walk's order, transcribed
from the kernels, its waits polls of the words the bridge stores, ended by a landing or a cancel (recorded in the
violation words, which end the program's later waits), so the Metal path's walk runs with no GPU kernel; the Metal 4
form's own steps are refused as the classic form refuses them). Its Metal half (the violation words, combine3, counted
all-to-all, split-KV partials, retain and contents) is the deployed library's, transcribed; the arithmetic is
nccl-mesh.c's own (`nccl_mesh_combine`, `nccl_mesh_premultiply`, `nccl_mesh_truncdiv`, hidden, where
`MESH_HOST_EXECUTOR` is defined).

**The driver** (`rdma/.build/host/mesh-loop`, rdma/mesh-loop.c; metal-microbench b9dbc43^ tools/nccl_demo.py `loop`'s
part that needs no Python). Every wait of the driver is on an event (a map's serial, a bridge's start, a process's exit,
a landing word, a pipe), none bounded by a clock but `revoke`'s, which places its revoke inside the wait:
- `mesh-loop up FAMILY N WINDOW_MIB DIRECTORY`: a line, ring, star, complete map or a torus of two rows, of N (2 to 8)
  nodes, every cable two loopverbs devices and two free loopback ports; each node's configuration written, its bridge
  started from its own copy of the loop bridge's binary (DIRECTORY/node<i>.bridge, staged and renamed into place;
  posix_spawn, its log in DIRECTORY), and the call returns once every bridge's map holds N nodes and its own cables are
  paired (`mesh_nodes_health`, `mesh_observe`), each bridge awaited on its map's serial (`mesh_map_serial`,
  `mesh_map_wait`), or on its start (`mesh_bridge_wait`) where it does not hold its region yet.
- `mesh-loop relay DIRECTORY BYTES`: every ordered pair, one after another, a program of one transfer of BYTES from the
  one to the other, its two ends agreeing their leases and the relays of their route (`mesh_transfers_agree`), the
  bridges between forwarding it where the two are not cabled; every word checked at the receiver.
- `mesh-loop nccl DIRECTORY COUNT host|walk`: N ranks, all-reduce, all-gather, reduce-scatter, broadcast and reduce
  from every root, and a send/receive around the ring, COUNT float32 each, every value exact, on the host path or the
  Metal path's walk on the host executor.
- `mesh-loop destroy DIRECTORY COUNT RUNS host|walk`: RUNS times, N ranks, an all-reduce then a last group (a send and
  receive around the ring) of COUNT float32 each, every value exact, every rank's `ncclCommDestroy` right after its last
  group; then every bridge read until its leases are free and its client slots vacant (`bridges_idle`), and the bridges'
  logs read: programs and relays ended in order, programs failed.
- `mesh-loop kill DIRECTORY COUNT host|walk [bridge|binary]`: N ranks, an all-reduce, then the last rank waits while the
  others begin a second and is killed (SIGKILL): each other rank's call fails as a value, the time from the kill to its
  return read; with `bridge`, every rank begins the second and the last rank's node's bridge is stopped (SIGTERM, as
  launchd stops one) in its place: every rank's call fails as a value, that node's rank by its watch of its bridge's
  exit (`mesh_watch`), the others by the FAREWELL (run it last: the stopped bridge stays stopped); with `binary`, the
  same with the last node's bridge's own binary replaced in its place (a new file renamed over DIRECTORY/node<i>.bridge,
  as an install does), which stops that bridge at once: every rank's call fails as a value, then the bridge is started
  again, as launchd's `KeepAlive` starts it, and the map awaited until every bridge's holds every node and its cables
  are paired.
- `mesh-loop sigterm DIRECTORY STATE [COUNT]`: every bridge sent SIGTERM in STATE and its exit awaited, as in
  [the stop](#waits-and-their-ends); the driver reaps its own exited children before it counts the fabric's rings held
  by a live process.
- `mesh-loop hop DIRECTORY ROUNDS`: nodes 0 and 1 (a cable), ROUNDS round trips of one float32 each way between two
  ranks (`ncclSend`, `ncclRecv` on the host path), the median one-way time of a round in microseconds.
- `mesh-loop share DIRECTORY COUNT`: two communicators over nodes 0 and 1 at once, each its own processes, their
  sessions opened one after the other, then both running the NCCL calls of `nccl` (COUNT float32) at once: both hold
  leases of the one cable.
- `mesh-loop revoke DIRECTORY COUNT host|walk`: N ranks; the last makes its communicator and never calls, and each
  other begins an all-reduce of COUNT float32, which waits for the last rank's part, while another of its threads
  revokes the communicator once the call has waited 100 ms (`ncclCommRevoke`, as metal-microbench's rank cancels a
  call): the call returns an error and `ncclCommDestroy` returns, and the time from the revoke to the call's return is
  read (a call not back 10 s after its revoke reads `hung`). Both paths' executors read the revoke themselves, so the
  reading does not reach the bridge's cancel of an aborted program's words, which a GPU walk's waits need (§4).
- `mesh-loop down DIRECTORY`: SIGTERM to every bridge and each exit awaited (`EVFILT_PROC`), the fabric unlinked.

The ranks of `nccl`, `destroy`, `kill`, `sigterm`, `hop` and `revoke` meet at a unique id the driver makes before it forks them
(`ncclGetUniqueId`: its root's listener opened in the driver and inherited, so every rank's connect meets it
listening); `share`'s make their communicators with `ncclMeshCommInitRank`, as a rank makes its own.

**Readings at mesh 2594a087** (MESH_VERSION 123; metal-microbench
output_data/dead-history-20261017/scratch-rewrite, all.txt and hop.txt, 256 MiB windows): on a line of 4, the NCCL
matrix at 4,096 elements 48 of 48 on the host path and 48 of 48 on the walk; relays 12 of 12 at 1 MiB; `destroy` 20
runs on the host path and 20 on the walk, 0 cancelled, 80 programs and 80 relays ended in order each, 0 programs
failed, the bridges idle after; `kill` 3 of 3 on the host path and on the walk (the survivors' calls failing as values
1.5-3.5 ms after the kill); the matrix 48 of 48 after; `kill ... binary` 4 of 4, the bridge back and the map converged,
the matrix 48 of 48 after; `kill ... bridge` 4 of 4. On a torus of 6 (two rows of 3): the matrix 96 of 96 on each path,
relays 30 of 30, `destroy` 20 runs each, 120 programs and 120 relays ended in order each, 0 failed; `kill` 5 of 5 on
each path, `kill ... binary` 6 of 6 with the bridge back, the matrix 96 of 96 after each, `kill ... bridge` 6 of 6.
`sigterm` as in [Waits and their ends](#waits-and-their-ends). `share` on a line of 3, three times: both communicators' 32 results exact each
time, no rank failed. `hop` on a line, 3,000 rounds, the median one-way time against MESH_VERSION 122 (5216474d, a
program's own queue pairs) in two alternating runs: on a line of 2, 15.3 and 12.7 us (122: 13.1 and 16.6); on a line
of 4, 26.3 and 26.0 us (122: 14.9 and 13.0), where each of the four bridges' paired ports spins two progress threads
on the one host's cores (12 threads for 3 cables) beside the two ranks.

**Readings at mesh 0b330488** (every lease's pair made again in place, a refused verbs call retiring the device, an
abort's words cancelled; the same tree, scratch-rewrite run15 to run17): the line of 4 and torus of 6 batteries as
above, unchanged (the matrix 48 of 48 and 96 of 96 on each path, `destroy` 20 runs each with 0 cancelled and every
program and relay ended in order, `kill`, `kill ... binary` and `kill ... bridge` as above); `sigterm` in all eight
states on a line of 4 twice and a torus of 6 once, 24 of 24 clean, 0.9-5.0 ms; `share` 3 of 3; `revoke` on a line of
3, three times on each path, 6 of 6 calls back 0.08-0.22 ms after their revoke and every `ncclCommDestroy` returned,
the matrix exact after; a ledger cut on a line of 2 read by rdma/ledger.py, 10 crossings each way paired by name; `hop`
on a line of 2 beside MESH_VERSION 122 back to back on a loaded host, 14.3-16.3 us against 15.4-20.0 us. Each bridge's
stop logged 10 queue pairs destroyed a port: the link's, made once.

**Readings at mesh a68717f9** (no verbs object torn down: a link's pairs let go of at its loss and made again in place
at its next pairing, a refused verbs call stopping the bridge, the stop exiting with no verbs call, loopverbs holding a
ring by its process; metal-microbench output_data/mesh-noteardown-20261018, line4.txt and torus6.txt, 512 MiB windows,
each reading's last lines kept): on a line of 4 and a torus of 6, `nccl` at 1,024 elements on each path, `relay` at 64
KiB, `revoke` and `share` with no rank failed; `destroy` 20 runs on each path, 0 cancelled, 80 programs and 80 relays
ended in order each on the line and 120 and 120 on the torus, 0 failed, the bridges idle after; `kill` of a client on
each path, of the binary (the bridge back) and of the bridge, every survivor's call failed as a value; `nccl` on each
path and `relay` again after the binary's restart, none failed; `sigterm` as in
[Waits and their ends](#waits-and-their-ends). `hop`, 200 rounds on one loaded host: 31.1 us median on the line of 4
(p10 18.8, p90 57.1) and 75.8 us on the torus of 6 (p10 40.1, p90 121.4). Each bridge's stop logged its 10 queue pairs
a port left to the kernel's reclaim at its exit.

**Reproduction targets** (800fe6bc^, its readings with the GPU walk on the host executor): the NCCL matrix on an 8-node
line 2,346 of 2,346; line 8, shuffled ring 8, star 8 and torus 2x4 104 of 104 each; ring 5 with routes 46 of 46. The
matrix itself (every datatype, op and root, metal-microbench b9dbc43^ tools/nccl_demo.py) is the consumer's driver's to
restore on these targets; `mesh-loop nccl` is its float32 core. The integration check: `mesh-loop up line 8 512 D`,
then `relay D 4194304` (56 pairs, the farthest 7 hops), `nccl D 65536 host` and `nccl D 65536 walk`, then the same on
star 8, ring 5 and torus 8; the bridges' relays have run on at most six nodes, on the loopback fabric alone.

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

