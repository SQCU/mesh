# The bridge: a node's resident service

Each node runs one bridge, `rdma/mesh-flow`, as the launchd job `io.mesh.bridge` (install.sh), kept running by launchd
(KeepAlive) and checked by the keeper. Its lifetime is the node's, not any call's. No consumer starts, stops or
signals it. It takes no arguments and reads no environment: what it serves is found on the node, and what a program
needs of it is asked for by the program.

The bridge's work, and nothing else:

- it finds its ports and the node each one reaches, and keeps each link paired (`#ports`);
- it keeps the mesh's map, its own record and every other bridge's, flooded over the links (`#ports`, `#region`);
- it serves every client at once, its node's processes and its neighbours, each with its own lease and programs
  (`#clients`);
- it pairs each program toward every node it binds, over a link or along the map's route, runs its transfers and ends
  them, and forwards the programs whose routes pass through its node (`#protocol`, `#flows`, `#routes`), over queue
  pairs and threads it holds a link and a port, not a program and a hop (`#lanes`, `#progress`, `#cables`);
- it tells a program's failure, naming the node, to the program and the peer (design/recovery.md).

Operator, 2026-10-14 (D10): "clients are clients, bridge services are bridge services. clients shouldn't know anything
about other clients on the same mesh. ... clients *ask*, bridges *do*". And (D3), on whether the bridge keeps links
paired with no client attached: "this emerges from other requirements. required."

The bridge is updated by the mesh's own update: install.sh replaces the installed binary only when it differs, and a
bridge whose binary changed stops on its own at its next turn (its programs fail as values), and launchd starts the
new one. SIGTERM (launchd's stop) ends it by the same teardown. A bridge never receives SIGKILL from the mesh
(RDMA-RULES.md); its teardown retires every verbs object in the order design/RDMA-KERNEL-RECOVERY.md gives.

## region

The node's one region is the shared memory object `/mesh0` (mesh.h `MESH_NAME`), made by the bridge at its start and
laid out (`mesh_layout`): an arena, the window, of the size its configuration states (the bridge's one operand, a
document its consumer writes in JSON, TOML or YAML, etc/bridge/README.md; no default, and a port whose device cannot
register it refused, logged and turned again later, design/recovery.md#registration; design/prepared-machine.md#M09, with the operator's words: the window is the bridge's, its sizing for
an execution the client's), a page-table row a block (its page entry, the mapping and address of the block a lease
bound to it; its buffer, the owner and the lease's extent; its delivery, the address and per-invocation stride of the
completion word a landing on it stores: 64 bytes, and a bit in each of the row planes), the node's ports (`MESH_PORTS`) and the client slots (`MESH_CLIENTS`).
The arena is the registered window: every port's device registers it whole when the device comes up,
before any queue pair exists (`device_up`), and deregisters it only when the device retires (`down_device`), so no
program's start or run changes a device's mappings while DMA may be live (the operator's rule, quoted at M09). Nothing
in the region is sized by a call: arena pages, page-table rows, transfer lists and publication targets are a lease,
taken as a program asks for them and refused as a value naming the execution where the window cannot give them, never
by growing the window.

A program's publications are its own: `mesh_transfers_prepare` derives them from its SEND lists, one target a slot of
each SEND transfer (the row it sends and its column of SEND cells), and lays them after its SEND cells in their one
lease, a section of the program's slot, sorted by row, which the slot names (`targets`, `target_count`); a producer's records (`mesh_publication_prepare`,
design/prepared-machine.md#M10) and the bridge's relay hand-off (#flows) find a row's run by bisection
(`mesh_row_targets`), and the slot's retirement reclaims them with its other leases. Until MESH_VERSION 111 every row
carried room for a publication toward each of `MESH_DESTINATIONS` destinations (1,088 of its 1,136 bytes), written as
a client bound the row: 1.11 GiB beside a 16 GiB window, resident as rows were leased, for the few rows a program
publishes; the region's tables are now 64.5 bytes a window page and 0.5 MiB besides (65 MiB at 16 GiB, 57 MiB at 14
GiB; design/prepared-machine.md#M09).

The map is the bridge's table of records, one a node (`struct mesh_record`): its node number, name and boot, the
record's sequence in that boot, its bridge's window in bytes, the node's memory in bytes and hardware model
(`hw.memsize`, `hw.model`), its ports by index, each port's peer, phase, code, bandwidth and pairings and the queue pairs
its device holds against its max_qp (`pairs`, `max_pairs`), and its health (mesh-plan.h `struct mesh_health`,
design/recovery.md#diagnosis): every node's window and model are in every bridge's map, read where the records are
(`#records`). Its
`map_nodes` records lie at `map_off` in arena pages the bridge leases for itself and leases again larger as the mesh
grows (`mesh_records`), so no node count caps it. The map serial (`map`) is the bridge's to bump (`mesh_notify`)
before and after each change, odd while it writes; its own record is in before the region opens. A reader copies the
records and keeps its copy where the serial it read before was even and is unchanged after (`mesh_map_whole`), the
seqlock's reader (Boehm, MSPC 2012): mesh-plan.h `mesh_map`, with `mesh_map_wait` for the next map. A record whose
links (each port's peer, phase and bandwidth), name, boot and facts are unchanged changes only its health, its
sequence and its ports' counts (codes, pairings, queue pairs), under the health serial (`health`), so the map serial
moves only when the mesh does; `mesh_nodes_health` reads the records under both. A change the map does not take (a
record that changes nothing it holds, or one the prune drops at once) moves neither serial.

The region's `bell` is the word the bridge's progress threads park on (`#progress`): a client rings it
(`mesh_transfers_wake`, a `mesh_notify`) as a group of its program starts, so a thread that parked while nothing moved
serves that group's first publication and landing at once.

A client slot (`struct mesh_client`) holds one client: its owner word (the client's generation, slot and process), its
execution's key, its program once started (`configured`), its ring depth and whether it is cyclic, its publication
targets once prepared (`targets`, above), and its programs by destination node, `MESH_DESTINATIONS` of them
(`mesh_destination`). Each is the destination's node, its `struct
mesh_tx` (SEND cells, cancellation words, transfer lists of `MESH_QUEUES` queues each way, trace) and its status
(`struct mesh_port_info`: phase, code, domain, `failed` the node where it failed, the destination or a relay on its
route, and `prepared`). A client writes its own slot's other words and its own rows; the bridge writes the statuses,
the ports and the map. No client reads, waits for or names another client's.

## ports

A port is an RDMA device of the node (`rdma_enN`, its interface `enN`), given an index in the order the bridge finds
it: at its start, and again when an interface attaches or its link comes on that no port has (`ports_found`). On its
turn (every 3 s, design/recovery.md#pause), a port without a link opens its device (a context, a protection domain
and, once its port is active, the whole window registered: design/prepared-machine.md#M09; checked against that
device's own max_mr on the port's own queue, a refusal logged with the extent, the errno and the extents held, and the
port turning again 60 s later, doubling to 30 min, the extents it took kept: design/recovery.md#registration), and
while its port is active it sends a hello (its node and boot) to every node on its link (the link-local all-nodes group, UDP
`MESH_PORT`). Of two nodes that hear each other, the lower node dials the higher's listener on the address
it heard, and the two exchange their link hellos (node, boot, the port's LID and GID, the block size) over that
connection, its control socket, which the bridge keeps with a keepalive. A port whose device is retiring takes no
hello and no connection until its turn has retired the device and opened it again. The link is then paired: the
region's port table names the peer node, the phase `MESH_PAIRED`, the port's bandwidth and its pairings. The table
holds no device name or address: those stay in the bridge, and a reader names a port by its index (`mesh_observe`).

Each change to its port table is a new record of the bridge's node, its sequence one more, which the bridge sends as
RECORD over every paired port. A bridge that hears a record newer than the newest it has heard of that node (`#records`)
puts it in its map and sends it on over its other paired ports, and a port newly paired is sent every record the map
holds: OSPF's flooding of link-state records (RFC 2328 §13). Every bridge's map so holds every node's newest record,
with no registry and nothing written by hand. A reader takes the map
from mesh-plan.h (`mesh_map`): two nodes are linked where each one's record holds a port paired with the other, and
members the links leave apart are joined along the map's route through other nodes. No reader joins records itself.

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
unique id, a call's name), and the bridges pair a program with the members' programs of the same name. A client that
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
and started (`mesh_transfers_start`). The start takes the bridge's map as it stands: the bridge pairs the program toward
each node it binds, over a port paired with that node or along the map's route through other nodes' bridges, and the
start returns once every destination is paired, or with the failure that destination's status names with its node
(EHOSTUNREACH where the map has no route to the node; ETIMEDOUT past the bridge's own windows, design/recovery.md#first;
the others mesh-call.h lists). A later loss fails the program alike, and the next start reads the map again.
`mesh_detach`, or the client's process exiting, retires the slot (vacant at once, `mesh_retire`) and ends its program:
its rows (a program's, or the leases of a lease-owning slot) are closed; once no port holds a flow of the client the
bridge releases them and returns their pages to the window in the same turn (`mesh_retired_release`, then
`mesh_reclaim`'s `mesh_buffer_reclaim`), so the window's unleased bytes (`mesh_window`, node-info) count them free as
soon as the client is gone; a lease that finds no room reclaims again before it fails, as a backstop. The bridge watches each slot's process from the turn it first sees the slot (a kqueue
`NOTE_EXIT` on its pid, registered again for each new slot of the process, since the kernel keeps one watch per pid;
a pid already gone at that turn is retired then), and a process's exit retires every slot it holds, its lease slot
and its programs' slots alike; the liveness turn (every 3 s) retires any slot whose process is gone that the watch
missed.

## records

A record is newer than another of its node where its boot is later, or its boot the same and its sequence later. Each
bridge keeps, apart from the map, the newest record it ever heard of each node (its boot and sequence: RFC 2328 §13's
database, mesh-flow.c `struct known`), and takes and sends on a record only where it is newer than that, the map holding
the node or not. The map's prune (a node no paired path reaches leaves it, §14) drops a node's record but not its entry,
so a copy of a record the map dropped, still in flight on a cycle of survivors after its node stopped, is no longer
taken as fresh and sent on again: it ends at the first bridge that has heard it. A record no newer than the newest heard,
of a node the map no longer holds, is put again only where a paired path now reaches its node (the paths came back
before its node's next record), and sent on only then. An entry unheard for six health periods (60 s) whose node the
map no longer holds compares with nothing (§14's MaxAge), so a node whose clock went back is heard again; a full table
of 256 entries gives the longest unheard of the nodes the map does not hold its place.

At MESH_VERSION 112 the dedupe read the map alone: once the prune dropped a node, any copy of its last record was fresh,
put, dropped and sent on, so a node of two or more cables that stopped in a mesh whose survivors held a cycle (a ring,
a Studio K4, two cables between one pair) left its last record circulating for good, each pass moving every survivor's
map serial twice and so reopening every communicator's session at its next group (output_data/sixnode-20261015
substrate/flood_storm.py; at 113, metal-microbench output_data/bridge-20261015/flood_db.py).

A record carries its node's facts with its links: its bridge's window (the bytes its configuration registers), its
memory and its hardware model. They are the driver's to read from its own bridge's map, every node's at once,
rather than each node's node-info over ssh (mesh-stat `nodes`).

## protocol

Over a link's control socket the two bridges speak in fixed messages (`struct message`), each keyed by an execution's
key:

| message | sent when | the receiver |
|---|---|---|
| READY (queues and slots, the route, the stripe, whether the sender's end takes queue pairs of its own) | a program starts toward a node (a routed pair's lower end; its higher end once the lower's READY reached it); the route is the nodes from the receiver to it | where the route ends at the receiver, pairs with its client's program of the key toward the sender (one awaiting it is given it at once), else holds the flow (`#flows`); where the route goes on, makes a relay's two flows and sends READY on with the rest of the route; on a relay's existing flow (the far end's READY coming back), mirrors that direction and sends the READY on toward the route's start (`#routes`) |
| PAIRS (an exclusive flow's queue pair numbers and sequence numbers; a lane rider's index and generation in its port's table) | the flow paired: an exclusive flow's queue pairs made (INIT) | its queue pairs ready to receive (RTR) and its receives posted; or its records taking its lanes' headers (`#lanes`) |
| POSTED | its receives posted; a relay's once its partner heard the next hop's POSTED | its queue pairs ready to send (RTS), served by its port's progress threads: the flow runs |
| END | the program ended on this side | its flow fails (ECONNRESET, domain 4); a relay's goes on |
| FAIL (code, node) | a flow failed, at the node named | its flow fails (ECANCELED, domain 4) naming that node; a relay's goes on |
| RECORD (a node's record) | the sender's port table changed, a newer record reached it, or the port paired | takes it where it is newer than the newest heard of its node, and sends it on (`#records`) |
| LANES (the port's lanes' queue pairs and the frames a header may name; none and a code where they failed) | the port's lanes made, once the link paired | its lanes ready to receive and send, each listening for its first header: the lanes are up (`#lanes`) |
| FAREWELL | the bridge stops | the link's end is orderly |

No SEND of a program is posted before the peer's receives are (credit flow control holds a SEND until its receive is
posted, TN3205; POSTED orders the two). POSTED is end to end: a relay tells it toward a route's end only once its
partner heard it from the next hop, so a routed program's start returns once the far member's program has started and
every relay between has posted, as a direct program's does (MESH_VERSION 112's relay posted at once, so a routed
start returned before the far member had started and its first group failed at the library's silent bound). A
program waits for its peer's READY at most the peer window (600 s) and for its queue pairs to pair at most the pairing
window (30 s), design/recovery.md#first; past either it fails (ETIMEDOUT).

## flows

A program toward one node is a flow on each port its route crosses: its receive rings and SEND streams
(design/prepared-machine.md M04-M30), a ring and a stream a queue of its toward the node and a slot of the queue. A
direct program's flows take queue pairs of their own where both ends' devices have room for them (`exclusive`, a queue
pair a ring, with its own two completion queues, its receives posted ahead as today), and ride their ports' lanes
otherwise; a routed program's flows, and every relay's, ride the lanes of the ports they cross (`#lanes`). A client's
flow runs over the client's own rows. On a node the route passes through, the program is a relay's two flows, toward
the route's previous node and its next, over rows the bridge leases itself, the one's receive rows the other's SEND
rows: a record landed from the previous node releases the SEND cell that carries it on (design/prepared-machine.md
M08), so a relay copies nothing and no process of its node takes part. On the route's last node, while no client's
program of the key has started there, the bridge holds the flow: it pairs it (its queue pairs made where it takes its
own) and answers PAIRS at once, posts its receives (then POSTED) once that program starts and binds its rows, and fails
it toward the sender (ETIMEDOUT) where none starts within the peer window.

Each port's verbs work (opening, querying and retiring its device, making and destroying queue pairs and lanes,
registering memory) runs on that port's own thread, so a call blocked in the driver blocks that port alone, and the
bridge's event loop, its other ports and its clients' starts go on (design/recovery.md Q14).

A flow fails alone: its client's pending words toward the node cancelled (M12), its status `MESH_STOPPED` with the
failure's code and domain and the node where it failed, its queue pairs destroyed (a lane rider's records posted on its
lanes landed first, or the lanes gone), its peer told (FAIL), and on a relay the route's other flow failed with it. A
failed flow is never resumed; the next program is a new flow, on the map as it is then.

## lanes

A device holds 11 queue pairs (max_qp, RDMA-FIRST.md), and at MESH_VERSION 112 every flow took its own: two a
destination a communicator's session (its classes' queues, nccl-mesh.c `session_open`), on every hop of a route, both
ends of every relay. One six-member communicator then asked 18 of a middle Studio of a chain, 12 of a ring's busiest
port, 10 of every port of a star's hub (a second communicator, 14), and the call failed (ENOSPC) on any of them
(output_data/sixnode-20261015 substrate/qp_demand.py). A port's queue pairs are now its link's: `MESH_LANES` (2) lanes,
UC queue pairs made when the link pairs and told the peer (LANES), which every flow on the link that takes no queue
pairs of its own shares, both ways, a flow's queue k riding lane k mod 2 (the library's large and small classes apart).
A direct program's flow takes queue pairs of its own only where both ends' devices have room for them beside the lanes
and every other flow's (`flow_make`: the device's max_qp less what its flows and lanes hold or will; each end's READY
or PAIRS tells its word, ANDed into the other's until the flow pairs, so both pair alike), and rides the lanes where
either has none: nothing is refused for a map's size or a node's communicators, and the pair's direct program keeps
its queue pairs.

TB5 has SEND and RECEIVE alone (no RDMA WRITE, no SEND with immediate: TN3205), and a queue pair's receives are taken
in the order posted, so rings that advance apart cannot share one by posting ahead. A lane's sender therefore heads
what it sends: each header (one 4 KiB frame) names, entry by entry, the records the SENDs after it fill (the receiving
flow by its index and generation in its port's table, its ring, the chunks and their bytes), and the receiver, on the
header's landing, posts those records of those rings in that order, then listens for the next header. Credit flow
control holds each SEND until its receive is posted (TN3205), so no SEND lands before its record and no record is
posted for a SEND that does not come; the data lands in the program's own rows, never copied. A header's entries are
the streams of the lane's flows whose cells their producers released, in turn, as many chunks as the lane's frames
take beside those in flight and the peer's receive room (its queue pair's and its share of the receive completion
queue, less the header's frame), a cell's chunks spanning headers where they must. An entry naming a flow that is no
longer the one its index names, or no longer takes records (it stopped: its rows are no longer its own), is posted as
discards of the same frames, so the lane stays in step; an entry whose frames are not its ring's records' fails that
flow, a header out of sequence or past the room the lane told fails the lanes, and a lane's failure is its link's
(design/recovery.md#loss). A flow on the lanes is released once its records posted on them have landed, or the lanes
it posted them on are gone. A lane costs its receiver a reaction a header (one way plus the receive thread's turn) that
a ring posted ahead does not, which is why a direct program keeps its own queue pairs where its devices have room.

A port's lanes hold two queue pairs whatever its programs, routes and communicators. Over the review's seven
six-node cablings and 240 numberings each, one six-member communicator holds 4 queue pairs on the busiest device (its
lanes and its direct neighbour's exclusive flow; 6 to 18 at 112, refused on three cablings), and with four splits
beside it 8 (12 to 36 at 112, refused on all seven); no direct flow had to ride the lanes, and a fifth communicator
toward one neighbour would; the pair holds 4 where it held 2 (metal-microbench output_data/bridge-20261015
qp_lanes.py, qp_lanes.json). The lanes' protocol, modelled with relays, stopped programs and reused table indices on
chains, rings, stars and a Studio K4, delivers every chunk to its record and completes every program that was not
stopped, 480 of 480 runs; posting the next header's receive before a header's records, posting no discards, or
reading no generation each misdeliver or desynchronise a lane in most runs (lanes_model.py, lanes_model.json).

## progress

A port's flows and lanes are served by the port's two progress threads, its receives' and its SENDs', started at its
first flow or its lanes and kept for the bridge's life: one pass serves every exclusive flow in the port's table (its
completion queues, its streams) and the port's lanes (their headers, their landings, their SENDs and completions).
A flow enters the table when it is to be served and leaves it once neither thread reads it (an epoch each thread tells
at each pass). MESH_VERSION 112 started two threads a flow, spinning while the flow lived, so a session kept 2(N-1)
spinning, and four a relayed pair: 10 to 34 on a member of a six-node mesh and 50 to 70 on a relay hub, above the
rank's encoding threads (QOS_CLASS_USER_INTERACTIVE). A node now runs two a paired port (at most two a cable,
independent of members, communicators and routes), and they spin only while something moves: after MESH_IDLE_NS
(100 ms) of nothing a thread parks on the region's `bell`, looking again every MESH_IDLE_POLL_NS (1 ms) while a flow
is in its table or its lanes are up, and at once when the bell rings, which a client does as its group starts
(mesh-call.h `mesh_transfers_wake`), the bridge as a flow is listed or unlisted, its lanes change or it stops. The
ready words a producer releases have no descriptor to wait on (RDMA-FIRST.md "Sleeps"), so a group's own ring is what
ends the park without a poll's delay; within a group, and for 100 ms after, the threads spin as before.

## cables

Several cables between two nodes are several ports, each with its own lanes. A direct program is a flow on each, in
the cable order both ends read alike (`cable_rank`: the index the lower node gave each), flow j carrying the program's
invocations t with t mod n = j (its `stripe` of `stripes`), so a bulk class's positions run over every cable at once
(MESH_VERSION 112 put a program's queue q on cable q mod n, so the large class rode one cable and a third carried
nothing). A flow's receive ring posts its own invocations ahead by its share of the program's span, so a stripe count
that does not divide a cyclic program's invocations stripes it unevenly rather than refusing it. Both ends' READYs
carry their stripe, and a flow whose ends disagree (one end sees a cable the other does not yet) fails as a value
(EPROTO). A routed program's flow, and a relay's, rides the lanes of the cable of its hop that the pair's hash picks
(the execution and the two ends mixed, alike at every bridge), so a pair's routed programs spread over a hop's cables.
The map prices a pair of several paired cables at their bandwidth together where the library reads it (mesh-dataflow.c
`mesh_paired`).

## routes

A routed pair's route is its lower node's: the lower end's program takes the route of its own map (mesh-dataflow.c
`mesh_route`, the library's routing table, mesh-plan.h `mesh_routes`, the same table every bridge and member makes of
the same records) and sends its READY along it; the higher end's program waits for that READY (an await, bounded by
the peer window) or takes the flow it already left held, and its READY goes back along the relays the lower's made,
each mirroring the direction it carries and sending it on, so every relay of the route carries both directions on
one route whatever either end's map held at its start (MESH_VERSION 112 routed each end on its own map: two maps a
link flap apart left both programs held on different relays, and a route of two or more relays carried the far end's
direction to its first relay alone). A READY that ends on a port where this node's program of its key toward the
sender runs flows elsewhere (one end saw a cable the other did not) is refused and the program failed at once, both as
values (EPROTO), rather than left to the peer window. A relay's flows ride its ports' lanes, so a route costs the
relays it crosses no queue pairs; each relayed direction still leases its rows and a client slot of the relay
(`mirror`), sized by its program's ring.
