# The bridge: a node's resident service

Each node runs one bridge, `rdma/mesh-flow`, as the admin user's launchd agent `io.mesh.bridge`, which
`bin/mesh-user.sh CONFIGURATION [CHECKOUT]` writes and loads from the user's checkout (`~/mesh`), kept running by launchd
(KeepAlive) and checked by the keeper. Its lifetime is the node's, not any call's. No consumer starts, stops or
signals it. Its one operand is its configuration (etc/bridge/README.md), and it reads no environment.

The operator, 2026-10-18: "once we are done we are done with the bridge; almost no featue or responsibility could ever
be assigned to the bridge service from thereon." And the same day: "what features does the bridge need that aren't in
nccl exactly?"; "why do you need any 'channel messages' at all?"; "do you really need a protocol at all or does the
underlying rdma driver provide all of the protocol features you thought you needed?" The bridge is five things, and
the five-second rule:

1. **One owner of the RDMA device per node.** Nothing else on a node opens it (`#ports`).
2. **Memory pinned once.** The window is registered whole on every device as the device comes up; clients map the
   region and lease slices of it (`#region`, `#clients`).
3. **Links found and paired again while running.** HELLO (UDP) finds the peer on each Thunderbolt interface; each
   link's control socket carries LINK, FAREWELL and RECORD; a link pairs again on the peer's next HELLO (`#ports`,
   `#protocol`).
4. **The map.** RECORD floods each node's number, boot, links, window and available memory to every node; clients read
   it from their own node's region (`mesh.nodes()`, mesh-plan.h `mesh_map`). Nothing else rides it (`#records`).
5. **Lanes as streams.** Lane 0 carries the keepalive frame; lanes 2 to 9 are four communicator pairs, leased by
   clients at NCCL's init; a program runs on its communicator's leased lanes, striped over a node pair's cables
   (`#lanes`, `#flows`, `#progress`, `#cables`).
6. **Five seconds.** No DMA or network channel of the bridge is used longer than mesh.h `MESH_PLATFORM_NS` without
   something happening on it or its scope concluding (design/recovery.md#five-seconds).

Its wire is HELLO, LINK, FAREWELL and RECORD. It carries no request, runs no command, keeps no service and forwards no
program (`#routes`). A node's facts beyond its record are its own resident rank's, and each member node's own driver
starts its ranks' part of a call (metal-microbench docs/elastic.md#call-delivery); a call's communicators meet at
NCCL's own bootstrap (rdma/NCCL.md), and nothing else passes between nodes.

Operator, 2026-10-14 (D10): "clients are clients, bridge services are bridge services. clients shouldn't know anything
about other clients on the same mesh. ... clients *ask*, bridges *do*". And (D3), on whether the bridge keeps links
paired with no client attached: "this emerges from other requirements. required."

The bridge is updated by the mesh's own convergence (README.md Convergence; bin/mesh-converge.sh): the admin user's
agent `io.mesh.converge` builds the upstream revision in a stage, installs its images between calls once the node is
quiet, and runs bin/mesh-user.sh on them. A build installs nothing (`make all`; design/recovery.md#install). A bridge
whose binary changed stops at once, at its next liveness turn (`#stop`), its programs failing as values, and launchd
starts the new one. install.sh, the root layer, installs only `bin/mesh-stat`, the reader node-info runs, built beside
the admin user's running bridge. SIGTERM (launchd's stop) ends the bridge, its programs failing as values. A bridge
never receives SIGKILL from the mesh (RDMA-RULES.md); its verbs objects are released by the kernel at its exit, never
by a call (design/RDMA-KERNEL-RECOVERY.md "October 18").

## files

The bridge, `rdma/mesh-flow`, is built from two translation units over one internal header (rdma/Makefile):

- `rdma/mesh-link.c`: the ports, links and their lanes, the lanes' leases as the link sees them (`lanes_take`,
  `lanes_release`), the progress threads and the whole data path they run (flows' SENDs, every lane's completions and
  landings, the link's keepalive: one unit, so `port_pass` inlines them as the measured build did), the event loop
  with its five-second conclusions, the peer interfaces' terminations, the provider's faults (`#faults`) and the stop.
- `rdma/mesh-program.c`: programs and their flows on leased lanes, the client slots, and the records.
- `rdma/mesh-bridge.h`: their types, the state both read (defined in mesh-link.c) and the functions each calls of the
  other's; `rdma/mesh-verbs.h` the verbs calls on a device and its queue pairs.

The link lanes and their leases:

- **Lanes.** A link's queue pairs are made once at its pairing as its `MESH_LINK_LANES` (10) lanes. Lane 0
  (`MESH_LANE_KEEPALIVE`) is the keepalive's and lane 1 is reserved and unused. From lane 2 (`MESH_LANE_FIRST`) the lanes
  are communicators' pairs (`MESH_LANE_PAIRS`, 4), one lane for the large class and one for the small.
- **Lease words** (struct hdr `lanes[port][lane]`). A word is 0 when the lane is free. Otherwise it holds the owner
  word of the client slot whose program rides it (a communicator's session slot), which its client claims and
  releases with a compare-exchange on the word it observes: at the release an owner word becomes 0, or
  `MESH_LANE_DIRTY` alone where the bridge set it. The bridge sets `MESH_LANE_DIRTY` as a flow takes the lane and
  clears it once the flow's stop finds nothing posted on the lane (`#lanes`) or the link has paired afresh, so no
  client leases a lane on which a stopped program's receives are still posted; it lets go of the lease of a client
  gone (its DIRTY kept).
- **A program's lanes** toward a destination: struct mesh_tx `lane`.
- **A link's lanes made at its pairing:** struct mesh_link_info `lanes`.
- **Choosing lanes at a communicator's init** (mesh-call.h). `mesh_lanes_free` gives a rank's report of the pairs free
  toward each peer. `mesh_lanes_choose` is the rule every rank applies to the gathered reports: for each link, the
  lowest pair free at both ends. `mesh_lanes_claim` and `mesh_lanes_release` take and give back a rank's chosen pairs.
  NCCL's bootstrap carries the reports (rdma/NCCL.md "Communicators").
- **Done words** are never lowered (a max register), so a cancellation's `~0` stands against a landing that comes after
  it.

## region

The node's one region is the shared memory object `/mesh0` (mesh.h `MESH_NAME`), made by the bridge at its start and
laid out (`mesh_layout`): an arena, the window, of the size its configuration states (a document its consumer writes in
JSON, TOML or YAML, etc/bridge/README.md; no default; design/prepared-machine.md#M09, with the operator's words: the
window is the bridge's, its sizing for an execution the client's), a page-table row a block (its page entry, the
mapping and address of the block a lease bound to it; its buffer, the owner and the lease's extent; its delivery, the
address and per-invocation stride of the completion word a landing on it stores: 64 bytes, and a bit in each of the row
planes), the node's ports (`MESH_PORTS`) and the client slots (`MESH_CLIENTS`). The arena is the registered window: every
port's device registers it whole when the device comes up, before any queue pair exists (`device_up`), and never
deregisters it (the kernel does at the bridge's exit), so no program's start or run changes a device's mappings while
DMA may be live (the operator's rule, quoted at M09); a device that refuses it leaves its port down until its link next
comes on (design/recovery.md#registration). Nothing in the region is sized by a call: arena pages, page-table rows,
transfer lists and publication targets are a lease, taken as a program asks for them and refused as a value naming the
execution where the window cannot give them, never by growing the window. A destination's transfer lists are leased at
its first transfer with room for the entries one block holds (42 a list at 16 KiB), and leased again twice as large,
the entries copied and the old lease returned, when a list fills (`mesh_lists_grow`, at most `MESH_ORDERS` a list).

A program's publications are its own: `mesh_transfers_prepare` derives them from its SEND lists, one target a slot of
each SEND transfer (the row it sends and its column of SEND cells), and lays them after its SEND cells in their one
lease, a section of the program's slot, sorted by row, which the slot names (`targets`, `target_count`); a producer's
records (`mesh_publication_prepare`, design/prepared-machine.md#M10) and the bridge's hand-off of a landing to the SEND
cell of the row it lands on (`#routes`) find a row's run by bisection (`mesh_row_targets`), and the slot's retirement
reclaims them with its other leases. The region's tables are 64.5 bytes a window page and 0.5 MiB besides (65 MiB at
16 GiB, 57 MiB at 14 GiB; design/prepared-machine.md#M09).

The map is the bridge's table of records, one a node (`struct mesh_record`): its node number, name and boot, the
record's sequence in that boot, its bridge's window in bytes, the node's memory in bytes and hardware model
(`hw.memsize`, `hw.model`), its ports by index, each port's peer, phase, code, bandwidth and pairings, the queue pairs
its device holds against those it can use (`pairs`, `max_pairs`: its max_qp less one) and the lanes made at its pairing,
and its health (mesh-plan.h `struct mesh_health`: its boot and its available memory, sampled at most 10 s apart). Its
`map_nodes` records lie at `map_off` in arena pages the bridge leases for itself and leases again larger as the mesh
grows (`mesh_records`), so no node count caps it. The map serial (`map`) is the bridge's to bump (`mesh_notify`)
before and after each change, odd while it writes; its own record is in before the region opens. A reader copies the
records and keeps its copy where the serial it read before was even and is unchanged after (`mesh_map_whole`), the
seqlock's reader (Boehm, MSPC 2012): mesh-plan.h `mesh_map`, with `mesh_map_wait` for the next map. A record whose
links (each port's peer, phase and bandwidth), name, boot and facts are unchanged changes only its health, its
sequence and its ports' counts (codes, pairings, queue pairs), under the health serial (`health`), so the map serial
moves only when the mesh does; `mesh_nodes_health` reads the records under both. A change the map does not take (a
record that changes nothing it holds, or one the prune drops at once) moves neither serial.

The region holds a `bell` a port, the word that port's progress threads park on (`#progress`): a client rings the
bells of the ports its program's flows ride (`mesh_transfers_wake`, a `mesh_notify` each; the slot's `ports`, a bit a
port, which the bridge sets as it takes the program's flows) as a group of its program starts, so a thread that parked
while nothing moved serves that group's first publication and landing at once, and no other port's threads wake. The
region's `busy` word is the count of flows the bridge holds, 0 between calls, and `idle_since` the wall clock (ns) at
which it last came to 0 (0 while it is not): a node's idle test (the convergence's, bin/mesh-converge.sh, through
mesh-stat's `idle` and `idle_since`, which the library's reader mesh-plan.h `mesh_busy` reads). `stopping` is written
and read by nothing: a changed binary stops the bridge at once (`#stop`).

A client slot (`struct mesh_client`) holds one client: its owner word (the client's generation, slot and process), its
execution's key, its program once started (`configured`), its ring depth and whether it is cyclic, its publication
targets once prepared (`targets`, above), the digest of what its program is made from (`digest`, 0 for none:
mesh-call.h `mesh_transfers_digest`, which libnccl-mesh's agreement compares, `#protocol`), and its programs by
destination node, `MESH_DESTINATIONS` of them (`mesh_destination`). Each is the destination's node, its `struct
mesh_tx` (SEND cells, cancellation words, transfer lists of `MESH_QUEUES` queues each way, the link lanes its program
rides toward it, `lane`) and its status (`struct mesh_port_info`: phase, code, domain, `failed` the node where it
failed, and `prepared`, the owner once the bridge took or failed the program toward it). A client writes its own slot's
other words and its own rows. The bridge writes the statuses, `ports`, the port table and the map, and in a program's
own leased rows each SEND cell's work request, its lane's queue pair and its port (`sends_configure`) and each landing's
completion word (`#lanes`). No client reads, waits for or names another client's.

## ports

A port is an RDMA device of the node (`rdma_enN`, its interface `enN`), given an index in the order the bridge finds
it: at its start, once IOKit's registry is quiet (at most `MESH_PLATFORM_NS`: design/recovery.md#five-seconds B9b),
and again when an `en<N>` interface attaches or its link comes on that no port has (`ports_found`; B11: no other
interface enters the driver), once every quiet window has ended. The bridge is the only process on its node that opens
the device. A port opens its device on the port's own queue (`port_open`: a context, a protection domain and, once its
port is active, the whole window registered: design/prepared-machine.md#M09; checked against that device's own max_mr,
a refusal logged with the extent, the errno and the extents held, the port down until its link next comes on, the
extents it took kept: design/recovery.md#registration; a context that fails with its port active is forgotten with no
verbs call and the device opened afresh), never on a clock: at the start, at its link's coming on, at a hello heard
while it is not open (B16), and as a call it was concluded around returns (B12); never inside its quiet window (B9) or
while a call of any port is stuck (B12).

An open port with no peer sends a hello (its node and boot) to every node on its link (the link-local all-nodes group,
UDP `MESH_PORT`) every 3 s (design/recovery.md#pause), a paced emission that decides nothing. Of two nodes that hear
each other, the higher answers with its own hello and the lower dials the higher's listener on the address it heard;
the two exchange their link hellos (node, boot, the port's LID and GID, the block size) over that connection, its
control socket. The link is then paired, and its lanes made (`#lanes`): the region's port table names the peer node,
the phase `MESH_PAIRED`, the port's bandwidth and its pairings. The table holds no device name or address: those stay
in the bridge, and a reader names a port by its index (`mesh_observe`). A control socket never blocks (a message queues
behind what it has not taken, `port_say`; more than `MESH_UNSENT`, 64 MiB, unsent loses the link), and it is heard
within `MESH_PLATFORM_NS` or its link concluded (design/recovery.md#five-seconds B2, B3).

Each change to its port table is a new record of the bridge's node, its sequence one more, which the bridge sends as
RECORD over every paired port. A bridge that hears a record newer than the newest it has heard of that node
(`#records`) puts it in its map and sends it on over its other paired ports, and a port newly paired is sent every
record the map holds: OSPF's flooding of link-state records (RFC 2328 §13). Every bridge's map so holds every node's
newest record, with no registry and nothing written by hand. A reader takes the map from mesh-plan.h (`mesh_map`): two
nodes are linked where each one's record holds a port paired with the other.

A node is its bridge's node number, a 31-bit fold of the host's UUID (`gethostuuid`), and its name the host's. A cable
moved to another port, a cable added, a peer that restarted: each is a port that finds its peer again, with no
configuration (there is none) and no restart, and its next record carries the change to every map. Several cables
between one pair of nodes are several ports, told apart by the port index.

## clients

A process attaches with `mesh_attach(context, name)`: it maps the region and takes a vacant slot with one
compare-exchange, never retires another's and never waits for any other client; a slot whose process is gone is the
bridge's to retire, on its turn. Its rows and arena pages are claimed with a compare-exchange on each bitmap word that
expects them clear (`mesh_scan`), so two clients preparing at once never hold the same rows or pages. A slot is one
execution's program: `name` is its execution's, every member of one execution names it alike (an NCCL communicator's
name, a call's name), and the members' programs of one name ride the lanes their communicator leased. A client that
runs several programs takes a slot for each on its one attach (`mesh_program_attach(program, client, name)`, sharing
the client's mapping). An attach is refused (EBUSY) when every slot is held, and ESRCH when the node's bridge is gone.

A client's memory is its leases, the system call the bridge is to its clients. The operator, 2026-10-15: "the bridge's
job is to allocate essentially all of the memory on the entire device so that there are large already 'pinned'
buffers which are ready to receive surprise transfers in whatever order they happen from whatever peers and links are
operational ... ontologically it must be the responsibility of client programs to know how much allocation they need
... the bridge is there to behave like a system call". A lease is a section of the window (`mesh_section_create`):
rows the client's slot owns and whole arena pages, registered on every device since the device came up, so a lease
registers nothing and no lease changes a device's mappings. libnccl-mesh makes the process one client: one attach,
made at its first `ncclMemAlloc` or communicator, whose slot owns every lease the process takes (`ncclMemAlloc`) and
on which each communicator's session takes its program's slot (rdma/NCCL.md "Memory"). The client sizes its leases:
one the window cannot meet is refused naming the bytes asked and the pages free, and the window never grows. Weights,
KV and every buffer a peer reads are leases: "why would weights or kv be outside of the memory pinned pool? being
outside of it means that they cannot be zero copy passed to peers." libnccl-mesh's collectives copy every operand, a
lease or not, through the session's rings, so none crosses a caller's buffer without a copy yet; a program that binds
its own section of the window (`mesh_transfer_bind`) crosses with none. A lease the client returns
(`mesh_lease_return`, `ncclMemFree`) is reclaimed at once by the reclaim a retired client's rows take; a client that
exits or dies has its leases closed with its slot and returned by the bridge (below). A lease is an operand of the
region it came from: when the bridge stops, its region goes with it, and a process holding a lease of it has its next
lease, communicator, session and `ncclCommRegister` fail (`ncclRemoteError`), as `ncclMeshMemLive` does, until it has
freed them all; it then attaches to the next bridge's region, never holding leases of two (rdma/NCCL.md "Memory").

A program is bound (`mesh_transfer_bind`: a transfer names its member node and queue, never a port; the destination's
lists are leased at its first transfer), prepared (`mesh_transfers_prepare`, `mesh_host_inputs` or the Metal transport)
and started (`mesh_transfers_start`). The start hands the program to the bridge and waits on nothing
(design/recovery.md#five-seconds C1): the bridge takes it toward each node it binds on the cables to that node on which
the client holds its lease (`#flows`), or fails that destination as a value, its status naming the code and the node
(design/recovery.md#first). A later loss fails the program alike, and the next start reads the map again.
`mesh_detach`, or the client's process exiting, retires the slot (vacant at once, `mesh_retire`) and ends its program:
its rows (a program's, or the leases of a lease-owning slot) are closed; once no port holds a flow of the client the
bridge releases them and returns their pages to the window in the same turn (`mesh_retired_release`, then
`mesh_reclaim`'s `mesh_buffer_reclaim`), so the window's unleased bytes (`mesh_window`, node-info) count them free as
soon as the client is gone; a flow that left a lane with receives posted keeps them until the link's next pairing has
destroyed those queue pairs, so no receive posted into them is left on the device. A lease that finds no room reclaims again before it fails. The bridge watches each
slot's process from the turn it first sees the slot (a kqueue `NOTE_EXIT` on its pid, registered again for each new
slot of the process, since the kernel keeps one watch per pid; a pid already gone at that turn is retired then), and a
process's exit retires every slot it holds, its lease slot and its programs' slots alike; the liveness turn (every
3 s) retires any slot whose process is gone that the watch missed.

## records

A record is newer than another of its node where its boot is later, or its boot the same and its sequence later. Each
bridge keeps, apart from the map, the newest record it ever heard of each node (its boot and sequence: RFC 2328 §13's
database, mesh-bridge.h `struct known`), and takes and sends on a record only where it is newer than that, the map
holding the node or not. The map's prune (a node no paired path reaches leaves it, §14) drops a node's record but not
its entry, so a copy of a record the map dropped, still in flight on a cycle of survivors after its node stopped, is
not taken as fresh and sent on again: it ends at the first bridge that has heard it, and a node of two or more cables
that stops in a mesh whose survivors hold a cycle (a ring, a Studio K4, two cables between one pair) leaves no record
circulating (output_data/sixnode-20261015 substrate/flood_storm.py). A record no newer than the newest heard, of a node
the map no longer holds, is put again only where a paired path now reaches its node, and sent on only then. A full
table of 256 entries gives the longest unheard of the nodes the map does not hold its place.

A record carries its node's facts with its links: its bridge's window (the bytes its configuration registers), its
memory and its hardware model, its boot and its available memory (`kern.memorystatus_level` of its memory, what the
kernel can give without pressure). They are the driver's to read from its own bridge's map, every node's at once
(`mesh.nodes()`; mesh-plan.h `struct mesh_node_health`, `mesh_nodes_health`; mesh-stat's `nodes`), never probed over
another network. This node's record is sent anew on every change of its port table, at a rename, and with its
available memory at most every `MESH_HEALTH_NS` (10 s), and told again at its current sequence on every paired port
each liveness turn (3 s), which a receiver drops as no newer than its newest
(design/recovery.md#five-seconds B3).

## protocol

Over a link's control socket the two bridges speak in fixed messages (`struct message`), the bridge's whole wire:

| message | sent when | the receiver |
|---|---|---|
| HELLO | over UDP, every 3 s from a port with no peer, and the higher node's answer; then first on the control socket | the link paired (`#ports`); its lanes made (`#lanes`) |
| LINK (every lane's queue pair number and first sequence number) | the link's lanes made, once a pairing | its lanes ready to receive and send, the link up (`#lanes`) |
| RECORD (a node's record) | the sender's port table changed, its available memory sampled, a newer record reached it, the port paired, and every 3 s its own again | takes it where it is newer than the newest heard of its node, and sends it on (`#records`) |
| FAREWELL (why the sender stops) | the bridge stops | the peer's "release everything": the link lost at once, nothing posted toward the peer again (design/recovery.md#farewell) |

Any other kind, a message out of its order, or a header whose magic is not this bridge's is a protocol error (EPROTO),
and the link is lost. The magic is a base plus `MESH_VERSION` plus the wire's revision, so bridges of different
versions or revisions drop each other's hellos and never pair. A program sends no message: its communicator's members
ride the same lanes from the two ends of each link, each end posting its receives in its program's order and its SENDs
as their cells are released, and TN3205's credit holds a SEND until its receive is posted (`#flows`). What a handshake
would carry is the compiled program's own: both ends' pieces agree by construction (mesh_compile; mesh-plan.h
`mesh_program_advise` checks the projection and the priorities that make it deadlock-free,
design/collectives-dataflow.md#advisories), a digest travels in a session's first position as data (nccl-mesh.c's
agreement), and a program's end is told to no one: a peer's program ends by what ends it at its own end (its link's
loss, a SEND held five seconds, its caller's revoke as its words). No program waits on a peer's window or a pairing's.

## flows

A program toward one node is a flow on each cable to it on which its client holds the program's lease: its receive
rings and SEND streams (design/prepared-machine.md M04-M30), a ring and a stream a queue of its (at most two queues, of
one slot each: the communicator's lane pair, queue k on the lane its slot names, mesh.h struct mesh_tx `lane`). A
flow holds no queue pair and makes no verbs call: it is made on the event loop as its program starts
(`program_take`), its lanes held from then (the link's `holder`, their lease words DIRTY), its records and cells
prepared on its port's queue (`flow_job`), and it is listed in the port's table, where the receive thread posts its
rings' first records at its first look (`flow_arm`) and every later one as a record lands, and the send thread posts
its streams' SENDs as their cells are released, through each lane's gate (struct link `gate`). A client's flow runs
over the client's own rows. A program toward a node no cable reaches fails at once as a value (EHOSTUNREACH), and one
whose lease no cable carries (ENOLCK): the bridge forwards nothing (`#routes`).

Each port's verbs work (opening and querying its device, making, connecting, flushing and destroying its lanes) runs
on that port's own queue, so a call blocked in the driver blocks that port alone, and the bridge's event loop and its
other ports go on; a call out past `MESH_PLATFORM_NS` concludes its port around it (design/recovery.md#five-seconds
B12).

A flow fails alone: its client's pending words toward the node `~0` and its status `MESH_STOPPED` with the failure's
code and domain and the node where it failed, stored from the event loop at once, and its program's flows on its other
cables with it. Its stop (`flow_stop`, on its port's queue) then lets go of its lanes (`lanes_release`: the holders
cleared, the flow out of the port's table once neither thread reads it, a lane it left with nothing posted freed, and
its link concluded where it left one with requests posted: `#lanes`), and frees it; a flow so held is kept with its
rows until the link's next pairing. No peer is told. A served lane whose SENDs none retire for `MESH_PLATFORM_NS` fails
its holder (B4). A failed flow is never resumed; the next program is a new flow, on the map as it is then.

## lanes

A device holds 11 queue pairs (max_qp), 10 of them usable (RDMA-FIRST.md: the bridge counts max_qp less one,
`device_up`). They are its link's: `MESH_LINK_LANES` (10) UC queue pairs, the link's lanes, made once at each of its
pairings on two completion queues the device's context holds (made once a context and kept across pairings), and kept
until the next. Lane 0 carries the keepalive (`#keepalive`), lane 1 is reserved and unused, and from lane 2 the lanes
are communicators' pairs (a large class's lane and a small class's), leased at a communicator's init (`#files`), four
communicators a link at once. Each lane holds a tenth of the completion queues' entries (`share`), so no lane overruns
them, and the lanes share the two completion queues: a completion names its lane by its queue pair's number
(`link_lane`), and goes to the lane's keepalive or the flow that holds it (a lane with no holder drops what comes).
Beside its lanes a link holds one block of the window, its keepalive frames' source and sink (`link_reserve`:
`MESH_PORTS` blocks leased at the start, whatever the cables; a port the window could not give one takes it as its
link first pairs).

Every verbs call of the link is made on the port's queue, under the port's guard (no queue pair of the device is
posted on while another is made, connected, moved to the error state or destroyed: the progress threads park until the
change returns, design/RDMA-KERNEL-RECOVERY.md "The stripe writes of October 1"), none for a pairing already lost
(B10), and none inside the port's quiet window, five seconds from its last conclusion, but a concluded link's flush at
the end of its own conclusion's window (design/recovery.md#five-seconds B9, B6 (b)):

- **Pairing** (the peer's hello heard on the control socket, `link_turn`): the lanes of the link's last pairing moved
  to the error state and destroyed (d6dceac's `down_pair`, the completion queues kept and what they hold of those
  lanes dropped; a queue pair whose destruction fails kept for the next pairing, B16), or, where the context went with
  the peer, forgotten and the device opened afresh; then d6dceac's `verbs_up`: ten UC queue pairs in INIT, each lane's
  number and first sequence number told the peer in one LINK, and once the peer's LINK is heard, each lane ready to
  receive (RTR) and then to send (RTS), the keepalive's first receives posted (`link_connect`).
- **Loss** (the link off, the interface detached, the peer interface terminating, the socket's end, a FAREWELL, five
  seconds unheard): no verbs call. The lanes are served no more before the socket closes and left as they are, a job of
  the lost pairing still on the port's queue dropped as it ends (`link_lost`, the pairing's number); once the quiet
  window ends, a link that was connected has its lanes moved to the error state, so nothing stays posted on them, unless
  its interface went off, its peer interface terminated, a call of the port stuck or a fault was absorbed meanwhile
  (B6 (b)).
- **A failed completion** (an error status or a failed poll, on any lane: `link_failed`): no lane served, every
  holder's flow failed, and the link concluded at once (`link_turn` reads `broken`, `port_lost`): both ends pair it
  afresh on hearing each other, every lane made again. **A post refused**: its flow alone fails (on lane 0, the
  keepalive's lane: the link concluded, `lane_lost`).
- **A program's stop** (`lanes_release`, on the port's queue): a lane on which it left nothing posted (every receive
  its ring posted landed, every SEND retired) has its `MESH_LANE_DIRTY` cleared, and the next communicator that leases
  the pair posts on it at once. A lane it left with a request posted could still land a SEND in its rows, and no
  verbs call is made at a stop: the flow keeps its rows, and the link is concluded (`broken`, `port_lost`) unless it
  was lost already, so both ends pair it afresh by the pairing's own sequence (d6dceac's: every queue pair destroyed and
  made again, the sequence proven on both devices), after which the rows are released. A communicator's session runs
  from its first group to its destroy and its rings post ahead without end, so a communicator kept for the process's
  life (as NCCL programs keep theirs; metal-microbench's rank keeps a call set's) leaves nothing posted between calls,
  and its destroy (a failure, a revoke, the process's exit) concludes the link once.
- **A link job that fails** concludes the link, which pairs again on hearing the peer (B16); where its make's
  registration was refused, the port is down until its link next comes on (design/recovery.md#registration).

Program data never rides a header-described receive (the operator, 2026-10-16: "nobody gave permission for anything
but a pre-posted exclusive flow"; metal-microbench docs/principles.md section 4): a communicator's lane carries its
programs' receives posted ahead, in its stream order, the lane its own while its lease holds. A header-described
receive, posted only as a header landed, cost 2-3 % a step on the pair against a flow's own queue pairs (mesh f4ba8792;
metal-microbench docs/measurement.md 2026-10-16).

## progress

A port's flows and link lanes are served by the port's two progress threads, its receives' and its SENDs', started at
its link's first connect or its first flow and never joined (they end with the process: `#stop`). One pass serves every
flow in the port's table whose lanes the link serves (the send thread its streams, the receive thread its rings' first
records; `served` read once a pass, so no post lands on a lane its link no longer serves:
design/recovery.md#five-seconds B8) and, while the link serves any lane, the link's two completion queues: every lane's
completions, a program's landings and SEND retirements among them, and lane 0's keepalive frames. `port_pass`, the
flows' and the link's passes are one translation unit and inline as the measured build's did (`#files`).
A flow enters the table when it is to be served and leaves it once neither thread reads it (an epoch each thread tells
when it moved since its last tell: the two threads read `high` and `epoch` on one line at every pass, and a store of
each pass's epoch by both bounced that line between their cores at the pass rate, a coherence miss in every pass of the
latency path; mesh git 765a8590^ told it every pass). A node runs two threads a paired port (at most two a cable,
independent of members, communicators and routes), and they spin only while something moves: after MESH_IDLE_NS
(100 ms) of nothing a thread parks on its port's `bell`, looking again every MESH_IDLE_POLL_NS (1 ms) while something is
its to look for (`port_pass`: a flow it serves, or a lane served: a keepalive frame may land at any time), else every
MESH_PAUSE_NS (3 s), and at once when the bell rings: a client's group of a program on the port starting (mesh-call.h
`mesh_transfers_wake`), a landing that released one of the port's SEND cells while its send thread was parked
(`send_wake`: an index hand-off, the cell's `port` naming whose thread posts it), the bridge listing or unlisting a
flow, its link's lanes changing. A thread woken by its bell with nothing to look for parks again at once. The ready
words a producer releases have no descriptor to wait on (RDMA-FIRST.md "Sleeps"), so a group's own ring is what ends
the park without a poll's delay; within a group, and for 100 ms after, the threads spin. A peer's landing on a parked
receive thread waits for its next look, at most 1 ms. A send thread tells it is parked, then looks once more before it
waits, and `send_wake` orders its release before it reads that word, so one of the two sees the other.

While the port's guard stands (a verbs change of its device), each thread tells the guard it holds at and parks on the
guard word until it is lowered (design/recovery.md#five-seconds B13), so a guarded call in the driver costs no core;
the raiser parks until both have told, at most `MESH_PLATFORM_NS`, past which the change is not made and the port is
concluded around it (B12).

## stop

A SIGTERM (launchd's stop), any other catchable signal whose default ends the process (each one's kqueue filter added
before its default is ignored, so none is lost: design/recovery.md#five-seconds B15), or a changed binary (read at
each liveness turn) stops the bridge from any state: FAREWELL queued on every control socket and the socket closed (the
kernel sends what it holds); the progress threads told to end; every program's words `~0` and its status the stop's
code (ESHUTDOWN), stored from the event loop's thread as plain memory stores (`programs_stop`); `bridge_pid` 0 and the
clients' control word notified; then the process returns once every port's quiet window and a terminating peer
interface's window have ended, at most `MESH_PLATFORM_NS` past the stop's start, reading the kernel's link events
meanwhile so a link going off extends its port's window (B9, B9b): the exit's reclaim never falls inside the
termination a conclusion may follow. Nothing at the stop waits on a port's queue, a progress thread or a verbs call,
and the bridge releases no verbs object: the kernel releases every object and thread at the exit
(design/RDMA-KERNEL-RECOVERY.md "October 18").

## faults

Apple's provider posts by a store to the device's doorbell mapping, which the driver unmaps as a peer interface
terminates; a post in that gap faults in `tbt_post_recv` (design/RDMA-KERNEL-RECOVERY.md's signature), and an exit
then is a reclaim inside the termination. So the bridge absorbs that one fault (design/recovery.md#five-seconds P4;
`fault_heard`, installed before any device is opened): a SIGSEGV or SIGBUS whose pc lies in the provider's text, on a
thread that posts on a port's queue pairs (its progress threads, or its link job's), maps one anonymous page at the
faulting page by hint (never `MAP_FIXED`, so nothing mapped is replaced), so the store lands in that page and no
doorbell rings, and the port is halted and then lost as at its link's going off (ENETDOWN, no verbs call). The bridge
does not exit. Any other fault takes the default action and ends the process.

## keepalive

Lane 0 carries the link's keepalive frame whenever the link is up, at both ends (design/recovery.md#five-seconds B1):
a 16-byte SEND of the link's block, one in flight at a time (`link_alive`, wr_id `LINK_ALIVE_SENT`, taken by
`link_send`), every `keepalive_ns` while a flow of the port is listed, where the configuration states one
(`keepalive_us`, etc/bridge/README.md; the bridge reads `keepalive`, nanoseconds), else every `MESH_PLATFORM_NS`/5
(1 s). A frame of the listed pace is movement, so the port's threads pass on at that pace; the idle beat is none, so
they park as they would without it. The receive side keeps up to 256 receives of one frame posted on lane 0 from the
lane's connect (`link_alive_listen`), its receive thread posting each again as it lands (`link_receive`, wr_id
`LINK_ALIVE_HEARD`). The frames heard are the link's proof of life that the event loop reads (`port_silent`: B2). Per
flow, at 12f484f8, the listed pace was measured; per link it is unmeasured.

Why the listed pace: on the pair (metal-microbench output_data/split-20261017, 2026-10-17) the Mac mini's Thunderbolt
RDMA path, quiet for the 40-50 us of a layer's window, completed its next SEND 6.2 us after its post where the M5's
took 2.5, and signalled a frame the M5 sent while it computed up to 10.6 us later; with a frame of the peer's landed
10 us before, its SEND completed in 2.7 us (completion.py, contention.py: each crossing's two one-way times by how far
apart the members published). The provider's completions are a ring the controller writes and the poll reads in user
space (`tbt_poll_qp_send`, `tbt_poll_qp_recv`: a descriptor's bit 0x15), so the time is the controller's, not a parked
thread's: the send thread saw each released cell within 0.04 us. A host-path exchange of 3,072 bytes read 12.8-13.8 us
a round after gaps of 0-10 us, 15.1-15.6 after 20 and 16.5-18.8 after 40-160 (gapprobe2); UDP on the same cable did
not keep it awake (gapprobe3). With frames every 5 us on the lanes, through the one call in which they flowed, the
Mini's SEND completion read 1.8-1.9 us and the bridge-to-bridge hop 2.98 us median, 2.60 p10, 4.21 p90 against 6.50,
5.08 and 8.88 (ka3 against tc62), the step's median 3,904 us, p10 3,673, mean of the last 120 4,029.

Posting on a link's long-lived queue pairs while another queue pair of the device is destroyed, unguarded, crashes the
bridge in `tbt_post_recv` (a store to an unmapped page at a mapping's offset 8: design/RDMA-KERNEL-RECOVERY.md's
September signature of the provider's shared doorbell mapping; mesh b21ced65, 9e0b3a4a), so every verbs change of a
device runs under its port's guard, no queue pair posted on meanwhile (`#lanes`).

## cables

Several cables between two nodes are several ports, each with its own lanes. A program between two nodes is striped
over the cables on which its client holds its lease: its invocations t with t mod s = j are its stripe j of s
(`stripe`, `stripes`), flow j on cable j in the cable order both ends read alike (`cable_rank`: the index the lower
node gave each), so a bulk class's positions run over every cable at once. The leases are claimed alike at both ends
of each link, and libnccl-mesh's agreement compares the cables each end leased, so both ends stripe alike or fail as
values (design/recovery.md#five-seconds W3d). A flow's receive ring posts its own invocations ahead by its share of the
program's span, so a stripe count that does not divide a cyclic program's invocations stripes it unevenly rather than
refusing it. The map holds a link a cable between two nodes, each at the slowest of their paired ports' bandwidths
(mesh-dataflow.c `mesh_join`), so the pair carries a striped program at their count times the slowest cable.

## routes

The bridge forwards no program (metal-microbench output_data/structural-20261018/DESIGN.md decision 1): a program
toward a node no cable reaches fails at once (EHOSTUNREACH). A member whose route to another crosses a node is reached
through that node as a member rank of the communicator with no contribution (mesh-plan.h `mesh_compile`'s
`contributors`), whose program receives on the rows it sends, so a landing releases the SEND cell that carries it on
(the index hand-off, design/prepared-machine.md M08) and no process of the node copies anything; its lanes are leased in
the same bootstrap as every member's. A call whose route changes is a new membership: the next call renegotiates it
(the operator, 2026-10-18: "the clientspace should renegotiate a traversible tree reduction ... on the next
client-level-function-call even if the mid-flight operation fails").
