# The bridge: a node's resident service

Each node runs one bridge, `rdma/mesh-flow`, as the admin user's launchd agent `io.mesh.bridge`, which
`bin/mesh-user.sh CONFIGURATION [CHECKOUT]` writes and loads from the user's checkout (`~/mesh`), kept running by launchd
(KeepAlive) and checked by the keeper. Its lifetime is the node's, not any call's. No consumer starts, stops or
signals it. Its one operand is its configuration (etc/bridge/README.md), and it reads no environment: what it serves is
found on the node, and what a program needs of it is asked for by the program.

The bridge's work, and nothing else:

- it finds its ports and the node each one reaches, and keeps each link paired (`#ports`);
- it keeps the mesh's map, its own record and every other bridge's, flooded over the links (`#ports`, `#region`);
- it serves every client at once, its node's processes and its neighbours, each with its own lease and programs
  (`#clients`);
- it runs each program toward every node it binds on the link lanes its client leased toward that node, its receives
  posted ahead in the program's order and its SENDs as their cells are released, with no message per program
  (`#flows`, `#lanes`); queue pairs and threads are a link's and a port's, made at the link's pairing, never a
  program's (`#progress`, `#cables`); a node a program's route reaches through others is the members' to reach, a relay
  being a member rank with no contribution (`#routes`);
- it carries requests, its clients' to the services of any node and its neighbours' to its own node's, admitting each
  against its node's memory, and serves holds, its node's node-info and, where configured, commands (`#requests`);
- it tells a program's failure, naming the node, to the program (design/recovery.md); the peer's program ends by what
  ends it at its own end;
- where its configuration keeps them, it writes each flow's ledger, its crossings timed, at the flow's end (`#ledger`).

Operator, 2026-10-14 (D10): "clients are clients, bridge services are bridge services. clients shouldn't know anything
about other clients on the same mesh. ... clients *ask*, bridges *do*". And (D3), on whether the bridge keeps links
paired with no client attached: "this emerges from other requirements. required."

The bridge is updated by the mesh's own convergence (README.md Convergence; bin/mesh-converge.sh): the admin user's
agent `io.mesh.converge` builds the upstream revision in a stage, installs its images between calls once the node is
quiet, and runs bin/mesh-user.sh on them; a bridge whose binary changed stops at once, at its next liveness turn (0fb8eb54's
rule, #stop), its programs failing as values, and launchd starts the new one. install.sh, the root layer, installs only `bin/mesh-stat`, the
reader node-info runs, built beside the admin user's running bridge. SIGTERM (launchd's stop) ends the bridge, its programs
failing as values. A bridge never receives SIGKILL from the mesh (RDMA-RULES.md); its lanes' and devices' verbs objects
are released by the kernel at its exit, never by a call (design/RDMA-KERNEL-RECOVERY.md "October 18").

## files

The bridge, `rdma/mesh-flow`, is built from two translation units over one internal header (MESH_VERSION 118's
mesh-flow.c split in two, then WP1's link layer, metal-microbench output_data/structural-20261018/DESIGN.md):

- `rdma/mesh-link.c`: the ports, links and their lanes, the lanes' leases as the link sees them (`lanes_leased`,
  `lanes_release`), the progress threads and the whole data path they run (flows' SENDs, every lane's completions and
  landings, the link's keepalive and stripes; one unit, so `port_pass` inlines them as the measured build did), the
  event loop and the stop.
- `rdma/mesh-program.c`: programs and their flows on leased lanes, the stripes' requests, the records and the
  services.
- `rdma/mesh-bridge.h`: their types, the state both read (defined in mesh-link.c) and the functions each calls of the
  other's.

The link lanes are made and served at MESH_VERSION 124 (#lanes), and programs ride them (WP2); the communicators'
init that leases them is libnccl-mesh's (WP3, WP4):

- **Lanes.** A link's queue pairs are made once at its pairing as its `MESH_LINK_LANES` lanes. Lane 0 is the
  keepalive's and lane 1 the stripe writes'. From lane 2 the lanes are communicators' pairs, one lane for the large
  class and one for the small.
- **Lease words** (struct hdr `lanes[port][lane]`). A word is 0 when the lane is free. Otherwise it holds the owner
  word of the client slot whose program rides it (a communicator's session slot), which its client claims and
  releases with a compare-exchange: at the release an owner word becomes 0, or `MESH_LANE_DIRTY` alone where the bridge
  set it. The bridge sets `MESH_LANE_DIRTY` where a program left the lane with positions outstanding (a receive
  posted and not landed, a SEND posted and not completed), lets go of the lease of a client gone (its DIRTY kept), and
  clears every DIRTY as the link pairs afresh.
- **A program's lanes** toward a destination: struct mesh_tx `lane`.
- **A link's lanes made at its pairing:** struct mesh_link_info `lanes`.
- **Choosing lanes at a communicator's init** (mesh-call.h). `mesh_lanes_free` gives a rank's report of the pairs free
  toward each peer. `mesh_lanes_choose` is the rule every rank applies to the gathered reports: for each link, the
  lowest pair free at both ends. `mesh_lanes_claim` and `mesh_lanes_release` take and give back a rank's chosen pairs.
- **Done words** are never lowered (a max register), so a cancellation's `~0` stands against a landing that comes after
  it.

`status[d].prepared` and `stopping` are deleted along with their users (`stopping` is no longer written: a changed
binary stops the bridge at once, #stop).

## region

The node's one region is the shared memory object `/mesh0` (mesh.h `MESH_NAME`), made by the bridge at its start and
laid out (`mesh_layout`): an arena, the window, of the size its configuration states (the bridge's one operand, a
document its consumer writes in JSON, TOML or YAML, etc/bridge/README.md; no default, and a port whose device cannot
register it refused, logged and turned again later, design/recovery.md#registration; design/prepared-machine.md#M09, with the operator's words: the window is the bridge's, its sizing for
an execution the client's), a page-table row a block (its page entry, the mapping and address of the block a lease
bound to it; its buffer, the owner and the lease's extent; its delivery, the address and per-invocation stride of the
completion word a landing on it stores: 64 bytes, and a bit in each of the row planes), the node's ports (`MESH_PORTS`) and the client slots (`MESH_CLIENTS`).
The arena is the registered window: every port's device registers it whole when the device comes up,
before any queue pair exists (`device_up`), and never deregisters it (the kernel does at the bridge's exit), so no
program's start or run changes a device's mappings while DMA may be live (the operator's rule, quoted at M09). Nothing
in the region is sized by a call: arena pages, page-table rows, transfer lists and publication targets are a lease,
taken as a program asks for them and refused as a value naming the execution where the window cannot give them, never
by growing the window. A destination's transfer lists are leased at its first transfer with room for the entries one
block holds (42 a list at 16 KiB), and leased again twice as large, the entries copied and the old lease returned,
when a list fills (`mesh_lists_grow`, at most `MESH_ORDERS` a list): through MESH_VERSION 113 every
destination of every program leased room for 4,096 entries a list, 1.5 MiB, for the two to four transfers a session
binds.

A program's publications are its own: `mesh_transfers_prepare` derives them from its SEND lists, one target a slot of
each SEND transfer (the row it sends and its column of SEND cells), and lays them after its SEND cells in their one
lease, a section of the program's slot, sorted by row, which the slot names (`targets`, `target_count`); a producer's records (`mesh_publication_prepare`,
design/prepared-machine.md#M10) and the bridge's hand-off of a landing to the SEND cell of the row it lands on (#routes)
find a row's run by bisection
(`mesh_row_targets`), and the slot's retirement reclaims them with its other leases. Until MESH_VERSION 111 every row
carried room for a publication toward each of `MESH_DESTINATIONS` destinations (1,088 of its 1,136 bytes), written as
a client bound the row: 1.11 GiB beside a 16 GiB window, resident as rows were leased, for the few rows a program
publishes; the region's tables are now 64.5 bytes a window page and 0.5 MiB besides (65 MiB at 16 GiB, 57 MiB at 14
GiB; design/prepared-machine.md#M09).

The map is the bridge's table of records, one a node (`struct mesh_record`): its node number, name and boot, the
record's sequence in that boot, its bridge's window in bytes, the node's memory in bytes and hardware model
(`hw.memsize`, `hw.model`), its ports by index, each port's peer, phase, code, bandwidth and pairings and the queue pairs
its device holds against those it can use (`pairs`, `max_pairs`: its max_qp less one), and its health (mesh-plan.h `struct mesh_health`,
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

The region holds a `bell` a port, the word that port's progress threads park on (`#progress`): a client rings the
bells of the ports its program's flows ride (`mesh_transfers_wake`, a `mesh_notify` each; the slot's `ports`, a bit a
port, which the bridge sets as it takes the program's flows) as a group of its program starts, so a thread that parked
while nothing moved serves that group's first publication and landing at once, and no other port's threads wake. The
region's `busy` word is the count of flows the bridge holds, 0 between calls, and `idle_since` the wall clock (ns) at which it last came to 0 (0 while it is not): a node's idle test (the
convergence's, bin/mesh-converge.sh, through mesh-stat's `idle` and `idle_since`, which the library's reader
mesh-plan.h `mesh_busy` reads). Its `stopping` word is no longer written: a changed binary stops the bridge at once (#stop), and its readers go with
it (nccl.h `ncclMeshBridgeStopping`, WP3).

A client slot (`struct mesh_client`) holds one client: its owner word (the client's generation, slot and process), its
execution's key, its program once started (`configured`), its ring depth and whether it is cyclic, its publication
targets once prepared (`targets`, above), the digest of what its program is made from (`digest`, 0 for none:
mesh-call.h `mesh_transfers_digest`, `#protocol`), and its programs by destination node, `MESH_DESTINATIONS` of them
(`mesh_destination`). Each is the destination's node, its `struct
mesh_tx` (SEND cells, cancellation words, transfer lists of `MESH_QUEUES` queues each way, trace) and its status
(`struct mesh_port_info`: phase, code, domain, `failed` the node where it failed, and `prepared`, to be deleted with its
waiter, WP4), and the link lanes its program rides toward it (`lane`, #lanes). A client writes its own slot's other words and its own rows; the bridge writes the statuses,
the ports and the map. No client reads, waits for or names another client's.

## ports

A port is an RDMA device of the node (`rdma_enN`, its interface `enN`), given an index in the order the bridge finds
it: at its start, and again when an interface attaches or its link comes on that no port has (`ports_found`). A port
opens its device at the start and at its link's coming on, while it has no peer, never on a clock (`port_open`, on the
port's own queue: a context, a protection domain and, once its port is active, the whole window registered:
design/prepared-machine.md#M09; checked against that device's own max_mr, a refusal logged with the extent, the errno
and the extents held, the port down until its link next comes on, the extents it took kept:
design/recovery.md#registration; a context that fails with its port active is forgotten with no verbs call and the
device opened afresh). An open port with no peer sends a hello (its node and boot) to every node on its link (the
link-local all-nodes group, UDP `MESH_PORT`) every 3 s (design/recovery.md#pause), a paced emission that decides nothing.
Of two nodes that hear each other, the lower node dials the higher's listener on the address it heard, and the two
exchange their link hellos (node, boot, the port's LID and GID, the block size) over that connection, its control
socket, which carries no keepalive: its peer's loss is the kernel's to tell (its end closed, a reset, the link's going
off). The link is then paired, and its lanes made (#lanes): the region's port table names the peer node, the phase
`MESH_PAIRED`, the port's bandwidth and its pairings. The table
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
rather than each node's node-info over ssh: the library's reader copies them (mesh-plan.h `struct mesh_node_health`,
`mesh_nodes_health`) and mesh-stat prints them in each of its `nodes` (`window`, `memory`, `model`).

## protocol

Over a link's control socket the two bridges speak in fixed messages (`struct message`), the bridge's whole wire:

| message | sent when | the receiver |
|---|---|---|
| HELLO | the link's control socket is up | the link paired (`#ports`); its lanes made (`#lanes`) |
| LINK (every lane's queue pair number and sequence number; this end's largest stripe piece) | the link's lanes made, once a pairing | its lanes ready to receive and send, the link up (`#lanes`) |
| RECORD (a node's record) | the sender's port table changed, a newer record reached it, or the port paired | takes it where it is newer than the newest heard of its node, and sends it on (`#records`) |
| FAREWELL | the bridge stops | the link's end is orderly |
| CHANNEL_OPEN, CHANNEL_CONFIRM, CHANNEL_REFUSE, CHANNEL_DATA, CHANNEL_WINDOW, CHANNEL_SHUT, CHANNEL_CLOSE | a request's channel (`#requests`) | (DESIGN.md decision 2 moves them to a daemon: WP6) |

Any other kind is a protocol error (EPROTO), and the link is lost. A program sends no message: its communicator's
members ride the same lanes from the two ends of each link, each end posting its receives in its program's order and
its SENDs as their cells are released, and TN3205's credit holds a SEND until its receive is posted (`#flows`). What
the deleted messages (READY, PAIRS, POSTED, END, FAIL and the STREAM phases) carried is the compiled program's own: both
ends' pieces agree by construction (mesh_compile; mesh-plan.h `mesh_program_advise` checks the projection and the
priorities that make it deadlock-free, design/collectives-dataflow.md#advisories), a digest travels in a session's
first position as data (nccl-mesh.c's agreement), and a program's end is told to no one: a peer's program ends by what
ends it at its own end (its link's loss, its caller's revoke as its words). No program waits on a peer's window or a
pairing's (the 600 s and 30 s windows are deleted with the messages they bounded).

## requests

A node asks its own bridge, never a shell on another node (the operator, 2026-10-16: "why should a node do programming
by sshing into a terminal mode on a mesh peer instead of querying their own mesh bridge to ask 'hey what neighbors do i
have'?"). The bridge is the node's one daemon, as PRRTE's prted, Ray's raylet and the kubelet are each their node's.
Clients ask it about the mesh: `mesh_nodes_health` (#records) for every node's record, memory and health. They send
requests through it to the services of any node: `mesh_request`. ssh remains for administering the substrate alone
(installing or upgrading a bridge), as PRRTE uses it only to start its daemons.

A request is a channel (RFC 4254 section 5, the SSH connection protocol's channels; its `direct-streamlocal` channel
to a Unix socket on the far host is what `ssh node nc -U path` used). A client connects to its own bridge at
[::1]:MESH_PORT and sends a `struct mesh_request` naming the node, by its bridge number (its own too), and the service.
It reads the bridge's int32 answer: 0 once the far end is open, else an errno.

- EHOSTUNREACH: the map holds no path to the node.
- ENOENT: the node serves no such service.
- ECONNREFUSED: the service's socket refused.
- ETIMEDOUT: no answer within the pairing window, 30 s.

The connection then carries the client's bytes to the service and the service's back. Each direction ends with its
writer's half close (shutdown SHUT_WR), and a channel that fails anywhere resets the connection.

A channel's messages ride the links' control sockets beside the bridges' own (#protocol):

| message | the receiver |
|---|---|
| CHANNEL_OPEN (its key, origin and destination; the service; the sender's window) | at the destination, connects the service's socket and answers CONFIRM, or REFUSE with why; elsewhere, opens the next hop along its own map's route and answers once that hop does |
| CHANNEL_CONFIRM, CHANNEL_REFUSE | answers the hop the OPEN came from, and at the origin the client |
| CHANNEL_DATA | gives the bytes to the channel's other end |
| CHANNEL_WINDOW (bytes) | may send that many more bytes toward the sender |
| CHANNEL_SHUT | that direction ended: the other end is told once its queue is given |
| CHANNEL_CLOSE (code) | the channel failed: every other end is told (a socket reset, the next hop CLOSE) |

- **Keys.** A channel's key is its origin's bridge number and sequence, the same on every hop. A relay holds the
  channel as two ends, each a port, and needs no route of its own beyond the next hop.
- **Windows.** Each hop grants its sender a window (128 KiB), and grants bytes back only as they leave toward the
  other end (RFC 4254 section 5.2), so a hop holds at most a window each way. A socket end is read only while the
  other end takes at once what is read. 256 channels a node at 128 KiB each stay under a port's MESH_UNSENT (64 MiB),
  so no client can make the bridge drop a link.
- **Losses.** A link lost fails every channel crossing it, and each channel's other ends are told.
- **Registration.** A service registers its listening Unix socket (MESH_REQUEST_SERVE: its name and its socket's
  path) on a connection it holds open. The registration lasts while that connection does, and a newer one of the
  same name replaces it. The resident rank registers `rank` (metal-microbench rank.swift) and registers again
  whenever its bridge restarts.
- **Compatibility.** The kinds came at MESH_VERSION 115, unchanged then: an older bridge dropped them (an OPEN toward
  it timed out), and the region was untouched. MESH_VERSION 116 adds the health record's `available` and a request's
  `bytes` (#admission).
- **Streams.** The raw stream of MESH_VERSION 118 (a request of kind MESH_REQUEST_STREAM, a queue pair each end made for
  it and its STREAM messages) is deleted with the per-program queue pairs: a request of that kind is refused
  (EPROTO), and the one-way stream reading is a client program on a leased lane pair (rdma/mesh-stream.c,
  design/bridge-dataflow.md#streams).

Exercised on one node (metal-microbench output_data/bridge-20261016): a 3 MiB echo through a registered service in
38.6 ms; 40 concurrent channels of 1 B to 5 MiB, all correct, in 0.44 s; a reader slower than its writer held to the
window; 300 clients dropping mid-request, each failing as a value (logged) with no descriptor left open and the
bridge idle after.

### admission

A request names the memory it will take outside the window at its node (`struct mesh_request` `bytes`, carried in its
OPEN). The destination's bridge reads the node's available memory at that moment and refuses the channel (REFUSE,
ENOMEM) where the reading is short of the bytes. Available memory is what the kernel can give without pressure, the
measure jetsam keeps: kern.memorystatus_level percent of hw.memsize. Each refusal is logged with the bytes asked, the
memory available and the pressure. This is the kubelet's admission (a pod is not admitted on a node under memory
pressure), with the client stating the bytes: the operator's 2026-10-15 rule that a client knows how much it allocates
(#clients). A request that names none, a cancel's, is admitted at any pressure. The health record carries `available`
beside `free` (the kernel's free pages, which leave out the cache it reclaims): on the Mini with its rank resident,
0.27 GiB free against 5.3 GiB available (2026-10-16).

The bridge also watches its node. While the kernel's memory pressure is at warning or worse, or the available memory
is under a sixteenth of the node's (1.5 GiB on the Mini, 8 GiB on the M5), it logs the largest processes its user
runs, by physical footprint (`memory low: ...; largest: <name> (<pid>) <GiB>, ...`). It logs this as the condition
begins and every minute while it lasts. That names what holds memory outside the window, whether the holder is a
bridge client or not (the 2026-10-16 builders were not); the bridge never ends a process for it.

### exec

Where its configuration states `exec = true` (etc/bridge/README.md), the bridge runs its peers' commands as its user,
as PRRTE's daemons launch a job's processes and forward their output once ssh has started the daemons.

- **The request.** Its first line is a command, which the user's shell runs (`-c`) in the user's home. The rest of
  the request is the command's standard input.
- **The answer.** The command's standard output and error, then its wait status (int32, little-endian), once both its
  output has ended and its process has exited (`mesh.run` returns the exit code, or -signal, with the output).
- **Isolation.** Its process group is its own. It holds no descriptor but its standard three
  (POSIX_SPAWN_CLOEXEC_DEFAULT: none of the bridge's devices or sockets). Its environment is HOME, USER, LOGNAME,
  SHELL, TMPDIR, LANG and PATH=/usr/bin:/bin:/usr/sbin:/sbin; the shell's own startup file (zsh's .zshenv) adds to it.
- **Aborts.** A channel that fails while the command runs sends its process group SIGTERM, and the bridge reaps it. A
  client aborts by resetting its connection (closing it with SO_LINGER 0); its bridge notices a reset even after the
  client has ended its input.
- **Trust.** Any node on a cable to this one may run commands here once `exec` is served, as it may already write
  into the window.

The bridge also serves `nodeinfo`: a channel to it is a connection to the node's own node-info (io.mesh.nodeinfo,
[::1]:8100, else 8099), so a driver reads every node's node-info by its bridge number. It needs no address and checks
no identity.

### holds

The bridge serves `hold` itself: a request of one line, `<name> <the unix second it ends> <why>`, writes
~/.cache/mesh/hold/<name>, which the node's convergence reads (README.md Convergence). A line of `<name>` alone removes
that file. The answer is a line, `held <name>` or `released <name>`, else why not. A driver holds every node its calls
cross this way: a relay, a node with no checkout, and a node whose node-info it has not read are all reached alike.

## flows

A program toward one node is a flow on each cable to it on which its client holds the program's lease: its receive
rings and SEND streams (design/prepared-machine.md M04-M30), a ring and a stream a queue of its (at most two queues, of
one slot each: the communicator's lane pair, queue k on the lane its slot names, mesh.h struct mesh_tx `lane`). A
flow holds no queue pair and makes no verbs call: it is made on the event loop as its program starts
(`program_take`), its lanes held from then (the link's `holder`), its records and cells prepared on its port's queue
(`flow_job`), and it is listed in the port's table, where the receive thread posts its rings' first records at its first
look (`flow_arm`) and every later one as a record lands, and the send thread posts its streams' SENDs as their cells
are released, through each lane's gate (struct link `gate`). A client's flow runs over the client's own rows. A
program toward a node no cable reaches fails at once as a value (EHOSTUNREACH), and one whose lease no cable carries
(ENOLCK): the bridge forwards nothing (`#routes`).

Each port's verbs work (opening and querying its device, making its lanes, registering memory) runs on that port's own
thread, so a call blocked in the driver blocks that port alone, and the bridge's event loop and its other ports go on
(design/recovery.md Q14).

A flow fails alone: its client's pending words toward the node cancelled (M12), its status `MESH_STOPPED` with the
failure's code and domain and the node where it failed, and its program's flows on its other cables with it. Its stop
(`flow_stop`, on its port's queue) lets go of its lanes before anything else (`lanes_release`: the holders cleared,
the flow out of the port's table once neither thread reads it, and a lane it left with positions outstanding marked
DIRTY), and frees it; no verbs call is made, and no peer is told. A failed flow is never resumed; the next program is a
new flow, on the map as it is then.

## lanes

A device holds 11 queue pairs (max_qp), 10 of them usable (RDMA-FIRST.md: the bridge counts max_qp less one,
`device_up`). They are its link's: `MESH_LINK_LANES` (10) UC queue pairs, the link's lanes, made once at each of its
pairings on two completion queues the device's context holds (made once a context and kept across pairings), and kept
until the next. Lane 0 carries the keepalive (#keepalive), lane 1 the stripe writes (#stripes), and from lane 2 the
lanes are communicators' pairs (a large class's lane and a small class's), leased at a communicator's init (#files),
four communicators a link at once. Each lane holds a tenth of the completion queues' entries (`share`), so no lane
overruns them, and the lanes share the two completion queues: a completion names its lane by its queue pair's number
(`link_lane`), and goes to the lane's keepalive, its stripes, or the flow that holds it (a lane with no holder drops
what comes).

Every verbs call of the link is made on the port's queue, under the port's guard (no queue pair of the device is
posted on while another is made, connected or destroyed: #stripes' guard), and only on hearing the peer, never on a
failure:

- **Pairing** (the peer's hello heard on the control socket, `link_turn`): the lanes of the link's last pairing
  destroyed (d6dceac's `down_pair`, the completion queues kept and what they hold of those lanes dropped), or, where
  the context went with the peer, forgotten and the device opened afresh; then d6dceac's `verbs_up`: ten UC queue pairs
  in INIT, each lane's number and first sequence number told the peer in one LINK (with this end's largest stripe
  piece), and once the peer's LINK is heard, each lane ready to receive (RTR) and then to send (RTS), its first
  receives posted (`link_connect`).
- **Loss** (the link off, the interface detached, the socket's end, a FAREWELL): no verbs call. The lanes are served
  no more and left as they are for the next pairing, a job of the lost pairing still on the port's queue dropped as
  it ends (`link_lost`, the pairing's number).
- **A lane dirty** (a completion's error or a post refused on a lane, found by a progress thread; or a leased lane a
  program left with positions outstanding, `MESH_LANE_DIRTY`): served no more where it failed, and leased to no new
  program. No lane is made again while its link is up, and no message asks for it (WP1's rounds, a handshake and a
  phase, are deleted: the 09-08 record, AGENTS.md). Once nothing rides the link here (no lane leased or held, no stripe
  in flight, no channel on it: `link_idle`), the bridge closes the link's control socket (`link_turn`), and both ends
  lose the link and pair it afresh on hearing each other, every lane made again by the pairing's own sequence. Leases
  are claimed alike at the two ends of a link, so a link with nothing on it here has nothing on it there.
- **A link job that fails** leaves the port down until its link next comes on (no retry on a clock).

Program data never rides a header-described receive (the operator, 2026-10-16: "nobody gave permission for anything
but a pre-posted exclusive flow"; metal-microbench docs/principles.md section 4): a communicator's lane carries its
programs' receives posted ahead, in its stream order, the lane its own while its lease holds. The header lanes of MESH_VERSION 112 to 118, which posted a
crossing's receives only as its header landed, are deleted: on the pair they cost 2-3 % a step against a flow's own
queue pairs (mesh f4ba8792; metal-microbench docs/measurement.md 2026-10-16).

## progress

A port's flows and link lanes are served by the port's two progress threads, its receives' and its SENDs', started at
its link's first connect or its first flow and never joined (they end with the process: #stop). One pass serves every
flow in the port's table (the send thread its streams, the receive thread its rings' first records) and, while the
link serves any lane (`link.served`), the link's two completion queues: every lane's completions, a program's landings
and SEND retirements among them, lane 0's keepalive frames, lane 1's stripes (#stripes). `port_pass`, the flows' and the
link's passes are one translation unit and inline as the measured build's did (#files).
A flow enters the table when it is to be served and leaves it once neither thread reads it (an epoch each thread tells
at the end of the first pass that read it: the two threads read `high` and `epoch` on one line at every pass, and a
store of each pass's epoch by both bounced that line between their cores at the pass rate, a coherence miss in every
pass of the latency path; mesh git 765a8590^ told it every pass). A node runs two threads a paired port (at most two a
cable, independent of members, communicators and routes), and they spin only while something moves: after
MESH_IDLE_NS (100 ms) of nothing a thread parks on its port's `bell`, looking again every MESH_IDLE_POLL_NS (1 ms)
while something is its to look for (`port_pass`: a flow it serves, or a lane served: a stripe request may name the
peer, and a keepalive frame or a stripe land, at any time), else every MESH_PAUSE_NS (3 s), and at once when the bell
rings: a client's group of a program on the port starting (mesh-call.h `mesh_transfers_wake`), a landing that released
one of the port's SEND cells while its send thread was parked (`send_wake`: a relay's index hand-off, the cell's `port`
naming whose thread posts it), the bridge listing or unlisting a flow, its link's lanes changing, a stripe request
(`mesh_requests_wake`). A thread woken by its bell with nothing to look for parks again at once. The ready words a
producer releases have no descriptor to wait on (RDMA-FIRST.md "Sleeps"), so a group's own ring is what ends the park
without a poll's delay; within a group, and for 100 ms after, the threads spin as before. A peer's landing on a parked
receive thread waits for its next look, at most 1 ms. A send thread tells it is parked, then looks once more before it
waits, and `send_wake` orders its release before it reads that word, so one of the two sees the other.

## stop

A SIGTERM (launchd's stop) or a changed binary (at once, 0fb8eb54's rule) stops the bridge from any state: FAREWELL
queued on every control socket and the socket closed (the kernel sends what it holds); every program's words `~0` and
its status the stop's code (ESHUTDOWN), stored from the event loop's thread as plain memory stores
(`programs_stop`); `bridge_pid` 0 and the clients' control word notified; then the process returns. Nothing at the
stop waits on a port's queue, a progress thread or a verbs call, and the bridge releases no verbs object: the kernel
releases every object and thread at the exit (design/RDMA-KERNEL-RECOVERY.md "October 18").

## keepalive

A configuration's `keepalive_us` (etc/bridge/README.md; the bridge reads `keepalive`, nanoseconds) bounds how long a
link's send thread lets pass between frames on the link's lane 0 while any of its lanes is leased (a communicator rides
it). The receive side keeps up to 256 receives of one frame into the link's block posted on lane 0 from the lane's
connect (`link_alive_listen`), its receive thread posting each again as it lands (`link_receive`, wr_id
`LINK_ALIVE_HEARD`); the send thread posts a 16-byte SEND of the block once the interval passed since its last, one at
a time (`link_alive`, wr_id `LINK_ALIVE_SENT`, taken by `link_send`). It was a flow's own extra queue pair at
12f484f8; per link it is new, and unmeasured.

Why: on the pair (metal-microbench output_data/split-20261017, 2026-10-17) the Mac mini's Thunderbolt RDMA path, quiet
for the 40-50 us of a layer's window, completed its next SEND 6.2 us after its post where the M5's took 2.5, and
signalled a frame the M5 sent while it computed up to 10.6 us later; with a frame of the peer's landed 10 us before, its
SEND completed in 2.7 us (completion.py, contention.py: each crossing's two one-way times by how far apart the members
published). The provider's completions are a ring the controller writes and the poll reads in user space
(`tbt_poll_qp_send`, `tbt_poll_qp_recv`: a descriptor's bit 0x15), so the time is the controller's, not a parked
thread's: the send thread saw each released cell within 0.04 us. A host-path exchange of 3,072 bytes read 12.8-13.8 us
a round after gaps of 0-10 us, 15.1-15.6 after 20 and 16.5-18.8 after 40-160 (gapprobe2); UDP on the same cable did
not keep it awake (gapprobe3). With frames every 5 us on the lanes, through the one call in which they flowed, the
Mini's SEND completion read 1.8-1.9 us and the bridge-to-bridge hop 2.98 us median, 2.60 p10, 4.21 p90 against 6.50,
5.08 and 8.88 (ka3 against tc62), the step's median 3,904 us, p10 3,673, mean of the last 120 4,029.

The header lanes carried them first (mesh b21ced65, 9e0b3a4a): posting on their long-lived queue pairs while a flow's
were destroyed on the same device, unguarded, crashed the M5's bridge in `tbt_post_recv` (a store to an unmapped page
at a mapping's offset 8: design/RDMA-KERNEL-RECOVERY.md's September signature of the provider's shared doorbell
mapping). Every verbs change of a device now runs under its port's guard, no queue pair posted on meanwhile (#stripes).

## stripes

A client writes into a neighbour's window with no program at the neighbour: a stripe (the operator, 2026-10-01:
"MAKING K-MANY COMPUTERS INTO A SHARED ADDRESS SPACE WHERE EXTREMELY HIGH NUMBERS OF CPU CORES CAN ASYNCHRONOUSLY START
WRITING TO STRIPES OF THE REALLY BIG SHARED ADDRESS SPACE"; metal-microbench docs/acceptance-20261014.md A51: "The
bridge serves a neighbour's request over a link whether or not a local client has configured a matching program").
Any thread of any client, or a GPU, puts a request (the bytes at an offset of its node's window, the neighbour, a word
and a value) into a ring of its own that it registered with its bridge (mesh-call.h `mesh_requests_create`,
`mesh_write`); the bridge sends a descriptor and then the bytes, the neighbour's bridge posts its receives at the
same offset as the descriptor lands, and each bridge stores the value into its word as its side completes. A port's
stripes ride its link's lane 1, made and made again with the link (#lanes), served by the port's progress threads
(design/bridge-dataflow.md#stripes, which also holds the deviations from mesh 8fc27cbe, where they were first built and
from which they are restored, and the integration check). Descriptors and pieces share the lane, so the receiver
listens for one descriptor at a time, the next only once the last one's pieces have their receives posted. The
descriptor's receive posting is a header's, the one the bridge keeps: stripes are a client's writes into memory, never
a program's operands (#lanes).

The link's lanes live while it is paired, so their posts would cross every other queue pair's change on the device.
Every change of the device's verbs objects therefore runs with the port's guard raised, the progress threads making no
pass until it returns, and nothing is destroyed but at the link's next pairing or a lane made again: the answer to the
double deregistration and the panics that ended 8fc27cbe (design/RDMA-KERNEL-RECOVERY.md "The stripe writes of
October 1").

## ledger

Where its configuration states `ledger = true` (etc/bridge/README.md), the bridge keeps a ledger of each flow: Q5P's
bridge trace (M27, the per-call bridge's MESH_LEDGER, mesh git 5702ea5, deleted with that bridge in 800fe6bc), which
located a crossing's stages on the pair, restored on the resident bridge, so that a crossing's ~20 us a hop today
can be set against Q5P's ~6 us bridge to bridge stage by stage (metal-microbench docs/measurement.md 2026-10-15 and
2026-10-16; Q5P is metal-microbench 533958a, mesh 5702ea5). The configuration is its one switch; the bridge's start
line says "ledgers kept".

- **What it holds.** A flow's ledger is made on its port's queue before either side is prepared or the flow listed
  (`ledger_open`). For each SEND cell its send thread first saw released (`flow_send`): the cell, its stream's cycle,
  and when it was seen, when its first SEND was posted, when its last was, and when that SEND's completion was taken.
  A lane's completions come in the order posted, so the completion whose gate count reaches the cell's last SEND's is
  the cell's (its lane's events kept a lane each). For each receive record that landed (`record_landed`): its index, its cycle, and when its word
  was stored. Every time is the bridge's
  `ledger_ns` (CLOCK_UPTIME_RAW, ns: CLOCK_MONOTONIC reads whole microseconds on macOS), each node its own.
- **Its bound.** A side's events are one pass of the flow's cells or records, and every pass of a cyclic program's,
  up to MESH_LEDGER_EVENTS (65,536) each; past it the flow runs on and the ledger keeps the first. They are allocated
  as the side is prepared, 56 bytes an event sent and 16 a landing (4.5 MiB a flow at the bound), with the cells' and
  records' names (32 and 16 bytes each).
- **Its cost.** Off: a branch on the flow's ledger pointer where a cell is seen, where its SENDs are posted, at a SEND
  completion and at a landing. On: a clock read and a store into the ledger's own buffers; nothing allocated, logged
  or locked on a progress thread.
- **How it is written.** At the flow's stop (`flow_stop`, once neither progress thread reads the flow), on its port's
  queue: its header into the bridge's log, then a line an event, into the
  log where there are MESH_LEDGER_LINES (2,048) or fewer, else into a file beside the log,
  `ledger-<key>-<node>-<peer>-<stripe>-<ledger>-<UTC time>.jsonl`, which the header names (the file's first line is
  the header again). A flow held across executions (a program planned ahead, metal-microbench
  docs/placement.md#retention-and-protention-the-next-execution-planned-ahead) is written when it ends.

The lines, each a JSON object (in the log after its line stamp):

- the header, `{"ledger":id,"key":"<execution, hex>","node":n,"peer":p,"port":i,"near":a,"far":b,"stripe":s,
  "stripes":S,"exclusive":1,"queues":Q,"qps":Q,"lanes":[l0,l1],"invocations":T,"cyclic":0|1,"started":ns,"sends":n,"send_room":N,
  "landings":m,"landing_room":M,"file":"<name>"|null}`: `node` this bridge's, `peer` the link's other node, `near`
  and `far` the program's two end nodes as this end names them (the flow at the link's other end names them the other
  way round), `id` the bridge's number of the ledger, which its event lines carry;
- a SEND cell, `{"s":[id,queue,invocation,position,binding,cycle,chunks,seen,first,posted,completed,bytes]}` (`bytes`
  the transfer's, -1 where the cell was not named; design/bridge-dataflow.md section 6);
- a landing, `{"r":[id,ring,invocation,position,binding,chunk,chunks,cycle,landed]}`.

A cell and its records share a name across the link: queue (the peer's ring of the same index, queue k's slot j), the
program's invocation, the position among that invocation's transfers of the queue (in their lists' order, which both
ends' streams and rings take, so the hardware pairs them), and the cycle; `binding` is the transfer's binding as its
list names it. A time not reached is 0: a cell seen and not yet posted at the flow's end.

`rdma/ledger.py PATH...` reads two nodes' logs or ledger files of an execution, pairs each flow with its peer's
across the link (the same key and stripe, the nodes and the ends each the other's) and a cell sent with its last
chunk landed, and gives each direction's seen to first post (the send thread's pass and gate), first to last post,
last post to completion (the RC round trip), and the one-way bridge-to-bridge hop with the clocks' offset cancelled:
each crossing paired with one the other way, of the same name where the program crosses back there, else the nearest
in time, hop = ((landed_B - seen_A) + (landed_A - seen_B)) / 2, as Q5P's analysis took it (an offset so taken holds
the two directions' hops alike). Not restored: Q5P's third time of a landing (after its ring's next records were
posted) and its layout lines (`trace_layout`, `trace_binding`), whose work the names now do.

## cables

Several cables between two nodes are several ports, each with its own lanes. A program between two nodes is striped
over the cables on which its client holds its lease: its invocations t with t mod s = j are its stripe j of s
(`stripe`, `stripes`), flow j on cable j in the cable order both ends read alike (`cable_rank`: the index the lower
node gave each), so a bulk class's positions run over every cable at once (MESH_VERSION 112 put a program's queue q on
cable q mod n, so the large class rode one cable and a third carried nothing). The leases are claimed alike at both
ends of each link, so both ends stripe alike. A flow's receive ring posts its own invocations ahead by its share of the
program's span, so a stripe count that does not divide a cyclic program's invocations stripes it unevenly rather than
refusing it. The map prices a pair of several paired cables at their bandwidth together (mesh-dataflow.c
`mesh_paired`, `mesh_hop_cost`).

## routes

The bridge forwards no program (DESIGN.md decision 1, 2026-10-18): a member whose route to another crosses a node is
reached through that node as a member rank of the communicator with no contribution (mesh-plan.h `mesh_compile`'s
`contributors`), whose program receives on the rows it sends, so a landing releases the SEND cell that carries it on
(design/prepared-machine.md M08) and no process of the node copies anything; its lanes are leased in the same
rendezvous as every member's. A call whose route changes is a new membership: the next call renegotiates it (the
operator, 2026-10-18: "the clientspace should renegotiate a traversible tree reduction ... on the next
client-level-function-call even if the mid-flight operation fails"). The relays of MESH_VERSION 112 to 123 (READYs
carrying a route, a relay's two flows over a lease of its own, the higher end's await of the lower's route) are
deleted. Beside its lanes a node's bridge holds its link pages, a block and the stripes' descriptors (32 KiB) a port
the region has (`link_reserve`: `MESH_PORTS` 8 of them at its start, whatever its cables; a port the window could not
give them at the start takes them as its link first pairs), and its map's records.
