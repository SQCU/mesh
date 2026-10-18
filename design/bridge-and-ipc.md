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
- it pairs each program toward every node it binds, over a link or along the map's route, runs its transfers and ends
  them, and forwards the programs whose routes pass through its node (`#protocol`, `#flows`, `#routes`), over queue
  pairs and threads it holds a link and a port, not a program and a hop (`#lanes`, `#progress`, `#cables`);
- it carries requests, its clients' to the services of any node and its neighbours' to its own node's, admitting each
  against its node's memory, and serves holds, its node's node-info and, where configured, commands (`#requests`);
- it tells a program's failure, naming the node, to the program and the peer (design/recovery.md);
- where its configuration keeps them, it writes each flow's ledger, its crossings timed, at the flow's end (`#ledger`).

Its event loop waits on one kqueue with no timeout, every turn of it an event's: its sockets (the request listener,
the HELLO socket, the kernel's network events, each link's control socket, each request's), its clients' processes'
exits (EVFILT_PROC), its own binary (EVFILT_VNODE), the power sources' and thermal pressure's notify(3) descriptors,
its configuration's paced emissions (EVFILT_TIMER: the `hello` and `health` intervals, etc/bridge/README.md), and a
user event its own threads and queues trigger (`wake`): a job's end on a port's queue, a progress thread's report, the
clients' control word (`control_watch`, an address wait on it), a memory-pressure change (a dispatch source), a host
name configd changed (an SCDynamicStore notification). Nothing in it reads a clock to decide.

Operator, 2026-10-14 (D10): "clients are clients, bridge services are bridge services. clients shouldn't know anything
about other clients on the same mesh. ... clients *ask*, bridges *do*". And (D3), on whether the bridge keeps links
paired with no client attached: "this emerges from other requirements. required."

The bridge is updated by the mesh's own convergence (README.md Convergence; bin/mesh-converge.sh): the admin user's
agent `io.mesh.converge` builds the upstream revision in a stage, installs its images, and runs bin/mesh-user.sh on
them; a bridge whose binary changed (the file it started from written, replaced, renamed or removed: its kqueue's
EVFILT_VNODE on it) stops at once, as a signal stops it: each awaiting program fails as a value (ESHUTDOWN), each port
tells its peer FAREWELL and is lost, so every program on it fails as a value here and at its peers, and launchd starts
the new binary (KeepAlive).
install.sh, the root layer, installs only `bin/mesh-stat`, the reader node-info runs, built beside the admin user's
running bridge. SIGTERM (launchd's stop) ends the bridge by the
same teardown, its programs failing as values. A bridge never receives SIGKILL from the mesh (RDMA-RULES.md); its
teardown, once its programs have failed and its region tells it gone, retires every verbs object in the order
design/RDMA-KERNEL-RECOVERY.md gives, each port on its own queue, the ports side by side.

## region

The node's one region is the shared memory object `/mesh0` (mesh.h `MESH_NAME`), made by the bridge at its start and
laid out (`mesh_layout`): an arena, the window, of the size its configuration states (the bridge's one operand, a
document its consumer writes in JSON, TOML or YAML, etc/bridge/README.md; no default, and a port whose device cannot
register it refused, logged and turned again at its next memory-pressure change or link event,
design/recovery.md#registration; design/prepared-machine.md#M09, with the operator's words: the window is the bridge's, its sizing for
an execution the client's), a page-table row a block (its page entry, the mapping and address of the block a lease
bound to it; its buffer, the owner and the lease's extent; its delivery, the address and per-invocation stride of the
completion word a landing on it stores: 64 bytes, and a bit in each of the row planes), the node's ports (`MESH_PORTS`) and the client slots (`MESH_CLIENTS`).
The arena is the registered window: every port's device registers it whole when the device comes up,
before any queue pair exists (`device_up`), and deregisters it only when the device retires (`down_device`), so no
program's start or run changes a device's mappings while DMA may be live (the operator's rule, quoted at M09). Nothing
in the region is sized by a call: arena pages, page-table rows, transfer lists and publication targets are a lease,
taken as a program asks for them and refused as a value naming the execution where the window cannot give them, never
by growing the window. A destination's transfer lists are leased at its first transfer with room for the entries one
block holds (42 a list at 16 KiB), and leased again twice as large, the entries copied and the old lease returned,
when a list fills (`mesh_lists_grow`, at most `MESH_ORDERS` a list): through MESH_VERSION 113 every
destination of every program, a relay's two of every relayed direction among them, leased room for 4,096 entries a
list, 1.5 MiB, for the two to four transfers a session or a relay binds.

A program's publications are its own: `mesh_transfers_prepare` derives them from its SEND lists, one target a slot of
each SEND transfer (the row it sends and its column of SEND cells), and lays them after its SEND cells in their one
lease, a section of the program's slot, sorted by row, which the slot names (`targets`, `target_count`); a producer's records (`mesh_publication_prepare`,
design/prepared-machine.md#M10) and the bridge's relay hand-off (#flows) find a row's run by bisection
(`mesh_row_targets`), and the slot's retirement reclaims them with its other leases. Until MESH_VERSION 111 every row
carried room for a publication toward each of `MESH_DESTINATIONS` destinations (1,088 of its 1,136 bytes), written as
a client bound the row: 1.11 GiB beside a 16 GiB window, resident as rows were leased, for the few rows a program
publishes; the region's tables are now 64.5 bytes a window page and 0.5 MiB besides (65 MiB at 16 GiB, 57 MiB at 14
GiB; design/prepared-machine.md#M09).

The map is the bridge's table of records, one a node (`struct mesh_record`): its node number and name, the record's
sequence (which orders a node's records, `#records`), its bridge's window in bytes, the node's memory in bytes and hardware model
(`hw.memsize`, `hw.model`), its ports by index, each port's peer, phase, code, bandwidth and pairings and the queue pairs
its device holds against those it can use (`pairs`, `max_pairs`: its max_qp less one), and its health (mesh-plan.h `struct mesh_health`,
design/recovery.md#diagnosis): every node's window and model are in every bridge's map, read where the records are
(`#records`). Its
`map_nodes` records lie at `map_off` in arena pages the bridge leases for itself and leases again larger as the mesh
grows (`mesh_records`), so no node count caps it. The map serial (`map`) is the bridge's to bump (`mesh_notify`)
before and after each change, odd while it writes; its own record is in before the region opens. A reader copies the
records and keeps its copy where the serial it read before was even and is unchanged after (`mesh_map_whole`), the
seqlock's reader (Boehm, MSPC 2012): mesh-plan.h `mesh_map`, with `mesh_map_serial` (the serial now) and
`mesh_map_wait` (the next whole map: an address wait on the serial, with no bound, until it differs). A reader that
meets the serial odd waits on it alike, its bridge's notification after the write or its exit ending the wait. A
record whose links (each port's peer, phase and bandwidth), name and facts are unchanged changes only its health, its
sequence and its ports' counts (codes, pairings, queue pairs), under the health serial (`health`), so the map serial
moves only when the mesh does; `mesh_nodes_health` reads the records under both. A change the map does not take (a
record that changes nothing it holds, or one the prune drops at once) moves neither serial.

The region holds a `bell` a port, the word that port's progress threads park on (`#progress`): a client rings the
bells of the ports its program's flows ride (`mesh_transfers_wake`, a `mesh_notify` each; the slot's `ports`, a bit a
port, which the bridge sets as it takes the program's flows) as a group of its program starts, so a thread that parked
while nothing moved serves that group's first publication and landing at once, and no other port's threads wake. The
region's `busy` word is the count of the flows that carry a program (each not yet failed or released) and the
awaiting programs the bridge holds, relayed ones among them, 0 between calls: an observation (mesh-stat's `idle`, which
the library's reader mesh-plan.h `mesh_busy` reads, and mesh-loop's), which nothing waits on. A bridge posts the
notify(3) name `MESH_STARTED` (mesh-plan.h; a loop bridge's followed by its region's name) once its sockets listen and
its region is open, so a client that waits for its node's bridge blocks on that event (`mesh_bridge_wait`) rather than
trying again; a client that waits for the bridge to go (a client that holds sessions past their execution and serves
on across bridges) blocks on its process's exit (`mesh_exit_wait`, EVFILT_PROC), then on the next one's start.

A client slot (`struct mesh_client`) holds one client: its owner word (the client's generation, slot and process), its
execution's key, its program once started (`configured`), its ring depth and whether it is cyclic, its publication
targets once prepared (`targets`, above), the digest of what its program is made from (`digest`, 0 for none:
mesh-call.h `mesh_transfers_digest`, `#protocol`), and its programs by destination node, `MESH_DESTINATIONS` of them
(`mesh_destination`). Each is the destination's node, its `struct
mesh_tx` (SEND cells, cancellation words, transfer lists of `MESH_QUEUES` queues each way, trace) and its status
(`struct mesh_port_info`: phase, code, domain, `failed` the node where it failed, the destination or a relay on its
route, and `prepared`). A client writes its own slot's other words and its own rows; the bridge writes the statuses,
the ports and the map. No client reads, waits for or names another client's.

## ports

A port is an RDMA device of the node (`rdma_enN`, its interface `enN`), given an index in the order the bridge finds
it: at its start, and again when an interface attaches or its link comes on that no port has (`ports_found`). A port
turns when an event asks it (`wanted`): its finding, its link coming on or its interface attaching, a paired link of
it lost (it says HELLO again at once), the configuration's HELLO interval for a port with no peer (`hello`, a paced
emission: a HELLO lost on its link has no other event, and a port that never paired otherwise waits for its next
event), or a memory-pressure change after its window's registration was refused. On its turn a port without a link
opens its device (a context, a protection domain and, once its port is active, the whole window registered:
design/prepared-machine.md#M09; checked against that device's own max_mr on the port's own queue, a refusal logged
with the extent, the errno and the extents held, and the port turning again at its next memory-pressure change or link
event, the extents it took kept: design/recovery.md#registration; a device that fails otherwise is retired and opened
again at once, and where that fails too, at its next event), and while its port is active it sends a hello (its node)
to every node on its link (the link-local all-nodes group, UDP `MESH_PORT`). Of two nodes that hear each other, the
lower node dials the higher's listener on the address it heard (the higher answers each hello it hears, so the lower
hears it at once), and the two exchange their link hellos (node, the port's LID and GID, the block size) over that
connection, its control socket. A port whose device is retiring takes no hello and no connection until its turn has
retired the device and opened it again. The link is then paired: the
region's port table names the peer node, the phase `MESH_PAIRED`, the port's bandwidth and its pairings. The table
holds no device name or address: those stay in the bridge, and a reader names a port by its index (`mesh_observe`).

Each change to its port table is a new record of the bridge's node, its sequence one more, which the bridge sends as
RECORD over every paired port, as it does at an urgent change of its health and at the configuration's `health`
interval (a paced emission; none where it states none; design/recovery.md#diagnosis). A bridge that hears a record newer
than the newest it has heard of that node (`#records`) puts it in its map and sends it on over its other paired ports,
and a port newly paired is sent every record the map holds: OSPF's flooding of link-state records (RFC 2328 §13). Every bridge's map so holds every node's newest record,
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
(EHOSTUNREACH where the map has no route to the node; ESRCH where its bridge exits; the others mesh-call.h lists). It
waits on those events alone, an address wait on each destination's status with no bound: a peer whose program never
starts is waited for until the client's cancel or revoke (design/recovery.md#first). A later loss fails the program alike, and
the next start reads the map again. `mesh_detach` retires the slot in order: its owner word is left `MESH_ORDERLY`
until the bridge has read it (the turn after, when it vacates the slot), and each running flow of its program ends in
order (`#control`: every cell it released posted, those SENDs' completions taken, then released, its peer told
nothing). An abort (`mesh_retire`: a revoke, a cancel of the call in flight, nccl.h `ncclMeshCommCancel`, or the client's process exiting) vacates the slot at once and fails each
flow (FAIL). Either ends its program:
its rows (a program's, or the leases of a lease-owning slot) are closed; once no port holds a flow of the client the
bridge releases them and returns their pages to the window in the same turn (`mesh_retired_release`, then
`mesh_reclaim`'s `mesh_buffer_reclaim`), so the window's unleased bytes (`mesh_window`, node-info) count them free as
soon as the client is gone; a lease that finds no room reclaims again before it fails, as a backstop. The bridge watches each slot's process from the turn it first sees the slot (a kqueue
`NOTE_EXIT` on its pid, registered again for each new slot of the process, since the kernel keeps one watch per pid;
a pid already gone at that turn is retired then), and a process's exit retires every slot it holds, its lease slot
and its programs' slots alike.

A client watches its bridge alike (mesh-dataflow.c `mesh_watch`: a kqueue `NOTE_EXIT` on the bridge's pid, one thread
a process, over a mapping of the region it keeps, registered as the process first opens the region): at the bridge's
exit it marks the region's bridge gone (`bridge_pid` 0), cancels each of its own slots' programs' words (`UINT64_MAX`
where none landed, every one of a cyclic program's, as the bridge cancels a flow that never landed), fails each of
their destinations' statuses (ESRCH) with its start woken, moves the map's and the health's serials by 2 (their parity
kept: a reader that found the bridge alive and had not yet slept finds its word changed), and wakes the waiters on the
map, health and control words, so no wait of the process, a GPU program's on its landing words among them, outlives its
bridge.

## records

A node's records are ordered by their sequence, and of two of one sequence the one of greater bytes is the newer (RFC
2328 §13.1's checksum, `record_newer`). Each bridge keeps, apart from the map, the newest record it ever heard of each
node (RFC 2328 §13's database, mesh-flow.c `struct known`), and takes and sends on a record only where it is newer than
that, the map holding the node or not. The map's prune (a node no paired path reaches leaves it, §14) drops a node's
record but not its entry, so a copy of a record the map dropped, still in flight on a cycle of survivors after its node
stopped, is no longer taken as fresh and sent on again: it ends at the first bridge that has heard it. A record older
than the newest heard is answered by that newest, sent back on the port it came from (§13 step 8); one equal to it, of
a node the map no longer holds, is put again only where a paired path now reaches its node (the paths came back before
its node's next record), and sent on only then. A bridge's own sequence starts at 1 each time it starts; a record of
its own node newer than the one it holds (an earlier start's, flooded back to it as a link pairs, or answered by step
8) moves its sequence past it and floods its record again (§13.4), so a node whose bridge started again is heard again
by sequence alone. A full table of 256 entries gives its place to the first entry whose node the map no longer holds.
The prune is also a loss (`#control`): a program awaiting a node it drops fails as a value (EHOSTUNREACH, `#routes`).

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

Over a link's control socket the two bridges speak in fixed messages (`struct message`), each keyed by an execution's
key:

| message | sent when | the receiver |
|---|---|---|
| READY (queues and slots, the route, the stripe of the program's stripes, whether the sender's end takes queue pairs of its own, the digest of what the sender's program is made from) | a program starts toward a node (a routed pair's lower end; its higher end once the lower's READY reached it); the route is the nodes from the receiver to it | where the route ends at the receiver, pairs with its client's program of the key toward the sender (one awaiting it is given it at once), else holds the flow (`#flows`), or, where that program names another digest, fails both (ESTALE, below); where the route goes on, makes a relay's two flows and sends READY on with the rest of the route; on a relay's existing flow (the far end's READY coming back), mirrors that direction and sends the READY on toward the route's start (`#routes`); on a flow that already heard its READY, the peer's program of the execution made again (its earlier one ended in order and told nothing): that flow fails as the peer's end (ECONNRESET naming it, its words past their landed counts cancelled, so a session held on it fails its group in flight as a value and reopens at its next) and the READY is taken as a new one |
| PAIRS (an exclusive flow's queue pair numbers and sequence numbers; a lane rider's index and generation in its port's table) | the flow paired: an exclusive flow's queue pairs made (INIT) | its queue pairs ready to receive (RTR) and its receives posted; or its records taking its lanes' headers (`#lanes`) |
| POSTED | its receives posted; a relay's once its partner heard the next hop's POSTED | its queue pairs ready to send (RTS), served by its port's progress threads: the flow runs |
| FAIL (code, node) | a flow failed, at the node named, before it ran (a running flow's FAIL is its control channel's) | its flow fails (ECANCELED, domain 4) naming that node; a relay's goes on, its code kept; one of ESTALE fails the receiver's whole program alike (below) |
| RECORD (a node's record) | the sender's port table changed, a newer record reached it, or the port paired | takes it where it is newer than the newest heard of its node, and sends it on (`#records`) |
| LANES (the port's lanes' queue pairs and the frames a header may name; none and a code where they failed) | the port's lanes made, once the link paired | its lanes ready to receive and send, each listening for its first header: the lanes are up (`#lanes`) |
| FAREWELL | the bridge stops | every flow on the link is lost at once (its peer's bridge is going: ECONNRESET), as at the socket's end after it |

Every message of a flow names its stripe beside its key and ends (`struct message` `stripe`), since several stripes of
one routed program may share a port (`#cables`). No SEND of a program is posted before the peer's receives are (credit flow control holds a SEND until its receive is
posted, TN3205; POSTED orders the two). POSTED is end to end: a relay tells it toward a route's end only once its
partner heard it from the next hop, so a routed program's start returns once the far member's program has started and
every relay between has posted, as a direct program's does. A relay flow tries to tell it at the end of each of its own
jobs and as its partner hears the next hop's POSTED, whichever comes last (`flow_tell`). A program waits for its peer's
READY and for its queue pairs to pair with no bound: until they come, or a failure as a value (its peer's FAIL, a link
lost, the map's prune of a routed peer), or its client's cancel or revoke (design/recovery.md#first).

A program may name the digest of what it is made from (mesh-call.h `mesh_transfers_digest`, after it is prepared and
before it starts; a session of libnccl-mesh's, its communicator's, the digest of its members, map and routes that its
agreement compares, nccl-mesh.c `map_digest`), and each READY carries it (a relay's READY onward its lease's, which `mirror` names from the READY it
mirrors). A READY whose end's program of its execution names another digest is refused, and both programs fail at once
toward every destination they bind (ESTALE, naming the node whose program differs, in each destination's status:
`program_stale`), whether the end's program has taken the READY's flow, awaits it, or starts later with the READY held
for it (its acceptance compares every READY held for its execution first); each of their flows tells its peer FAIL with
ESTALE naming that node (the refusing node, toward the node it names), which fails that peer's program alike and is
never told back to its sender, so every member of an execution whose programs differ fails at start as a value rather
than one waiting for a READY no program of the other's takes. A program is compared only while it
runs (started, and not yet failed toward every destination it binds: `program_runs`); a READY that meets one that
ended or failed is held for the execution's next program, as one that meets none. A program that names none (0) is
compared with nothing. A message names a flow by its execution, its two ends and its stripe, not by its program's
generation, so a FAIL in flight from a program that failed can still fail its successor of the same execution where
that successor starts first; it fails as a value and runs again on the map as it is then.

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
- ENOMEM: the node's available memory is short of the request's bytes (#admission).
- the code of a link of its route lost before the far end answered (ENETRESET where it names none).

A channel unanswered waits for its answer, its route's loss or its client's end, with no bound.

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
  other end takes at once what is read.
- **Losses.** A link lost fails every channel crossing it, and each channel's other ends are told.
- **Registration.** A service registers its listening Unix socket (MESH_REQUEST_SERVE: its name and its socket's
  path) on a connection it holds open. The registration lasts while that connection does, and a newer one of the
  same name replaces it. The resident rank registers `rank` (metal-microbench rank.swift) and registers again
  whenever its bridge restarts.
- **Compatibility.** The kinds came at MESH_VERSION 115, unchanged then: an older bridge dropped them, and the region
  was untouched. MESH_VERSION 116 adds the health record's `available` and a request's
  `bytes` (#admission).
- **Streams.** A request of kind MESH_REQUEST_STREAM (MESH_VERSION 118; mesh-dataflow.h `mesh_stream`) asks the
  bridge for a raw one-way stream of so many bytes to a neighbour's bridge over a queue pair each end makes for it,
  every word checked there, its answer the two ends' numbers (design/bridge-dataflow.md#streams): the link's raw
  rate, read by the one process that opens verbs.

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
begins and again as the kernel's pressure changes while it lasts. That names what holds memory outside the window, whether the holder is a
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
- **Aborts.** A channel that fails while the command runs sends its process group SIGTERM, and the bridge reaps it at
  its exit (EVFILT_PROC). A client aborts by resetting its connection (closing it with SO_LINGER 0); its bridge takes
  the reset as that end's error at its next read or write of it.
- **Trust.** Any node on a cable to this one may run commands here once `exec` is served, as it may already write
  into the window.

The bridge also serves `nodeinfo`: a channel to it is a connection to the node's own node-info (io.mesh.nodeinfo,
[::1]:8100, else 8099), so a driver reads every node's node-info by its bridge number. It needs no address and checks
no identity.

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
own) and answers PAIRS at once, and posts its receives (then POSTED) once that program starts and binds its rows,
holding it with no bound until then, its peer's FAIL or its link's loss.

Each port's verbs work (opening, querying and retiring its device, making and destroying queue pairs and lanes,
registering memory) runs on that port's own thread, so a call blocked in the driver blocks that port alone, and the
bridge's event loop, its other ports and its clients' starts go on (design/recovery.md Q14).

A flow that fails fails its client program's every flow, toward every node: each is muted first on the event loop
(its port's progress threads settled, so no cell of the program is posted past this: `program_quiet`), then each has
its client's words toward its node cancelled past the landed count (M12, `#control`), its status `MESH_STOPPED` with
the failure's code and domain and the node where it failed, its peer told (FAIL: an exclusive flow's on its control
queue pair by its send thread, a lane rider's on its lanes, one not yet running on the control socket), and its queue
pairs moved to the error state and destroyed under the port's guard (a lane rider's records posted on its
lanes landed first, or the lanes gone); on a relay the route's other flow fails with it. A failed flow is never
resumed; the next program is a new flow, on the map as it is then. A flow whose program ended in order is released
(`#control`) and is not a failure.

## lanes

A device holds 11 queue pairs (max_qp), 10 of them usable (RDMA-FIRST.md: the bridge counts max_qp less one,
`device_up`), and at MESH_VERSION 112 every flow took its own: two a
destination a communicator's session (its classes' queues, nccl-mesh.c `session_open`), on every hop of a route, both
ends of every relay. One six-member communicator then asked 18 of a middle Studio of a chain, 12 of a ring's busiest
port, 10 of every port of a star's hub (a second communicator, 14), and the call failed (ENOSPC) on any of them
(output_data/sixnode-20261015 substrate/qp_demand.py). A port's queue pairs are now its link's: `MESH_LANES` (2) lanes,
UC queue pairs made when the link pairs and told the peer (LANES), which every flow on the link that takes no queue
pairs of its own shares, both ways, a flow's queue k riding lane k mod 2 (the library's large and small classes apart).
A direct program's flow takes queue pairs of its own only where they fit a PAIRS message (`MESH_PAIRS`, 16) and both
ends' devices have room for them beside the lanes and every other flow's (`flow_make`: the device's usable queue pairs
less what its flows and lanes hold or will, the lanes' two counted before they are made, so no flow's queue pairs
leave the lanes without room; each end's READY or PAIRS tells its word, ANDed into the other's until the flow pairs,
so both pair alike), and rides the lanes where either has none: nothing is refused for a map's size, a node's
communicators or a program's queues and slots (a lane rider holds no queue pair a ring), and the pair's direct program
keeps its queue pairs. A program's operands ride exclusive flows alone, their rings posted ahead (the operator,
2026-10-16: "nobody gave permission for anything but a pre-posted exclusive flow"; metal-microbench docs/principles.md
section 4): a lane's receiver posts a crossing's records only on its header's landing, a wire round trip on every
crossing, so the lanes are no path for a program's data (e500528c put the pair's direct program on them, 20.2 us a
crossing; 921312ce took it off). The making of a flow, the lanes' reason, is paid before the execution that uses it,
on a posterior over what that execution will be (metal-microbench docs/placement.md#retention-and-protention-the-next-execution-planned-ahead);
until that estimator places the device's queue pairs, a routed program and one whose queue pairs do not fit still
ride the lanes, which is a defect to remove, not a form. The lanes' pages (their headers and discard blocks, 7 pages at 16 KiB) are leased for every
port the region has at the bridge's start, beside the map's (`lanes_reserve`), so a port found later, once clients
hold the window, has them; a port the window could not give them takes them as its lanes are made, and waits (logged
once) where the window has none, as lanes wait for a device's room.

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
flow, a header out of sequence or past the room the lane told fails the lanes. Lanes that failed, or could not be made,
are reset, never the link (`lanes_reset`, design/recovery.md#loss): the flows that posted on them fail, the peer is told
(a LANES of none and why) and resets its own where they were joined to these, and both make new lanes at once, the link
paired and its exclusive flows running throughout; a flow still pairing waits for them. Lanes that could not be
connected are the link's loss, its device retired and opened again as a new bridge would open it (a context that
refuses every connection, as after a peer's restart, is mended so), and the link pairs again. A peer's LANES that
names new queue pairs while this end's lanes are joined to its old ones resets them alike (the TCP stream orders a peer's LANES of none before its new ones). Lanes are up (their `up` word, which the
progress threads read) once the event loop takes their connect's end, never in the connect itself, so lanes reset while
they connected stay down (through MESH_VERSION 114 the connect's job wrote `up` as it ended, after a reset the event
loop had made meanwhile, and the progress threads served lanes joined to queue pairs the peer had destroyed until
their teardown). A flow on the lanes is released once its records posted on them have landed, or the lanes' queue
pairs that held them are destroyed (their generation moved): through MESH_VERSION 114 also once the lanes were merely
down, so a relay's lease could be returned to the window, and leased by another client, while its records were still
posted on lanes not yet torn down and a peer's SEND in flight could land in it.

A lane costs its receiver a reaction a header that a ring posted ahead does not: the header's frame (4 KiB, 0.46 us
at the link's 8.9 GB/s, RDMA-FIRST.md), the receive thread's turn and the records' posts, with the header's SENDs held
by credit flow control until then (the resumption of a held SEND is unmeasured). An estimate of at least 1.5 us a
crossing: on the pair's coded ledger call (71 crossings a step at 17.26 ms, metal-microbench docs/measurement.md
2026-10-15) 0.11 ms a step, 0.6 %; at the form the engine is building, Q5P's (141 crossings a step at 3.96 ms, each
landing 5 to 18 us after its publication) 0.21 ms, 5.3 % of the step and up to a third of a crossing. That is why a
direct program kept queue pairs of its own where its devices had room, a second transport beside the lanes for the
latency-bound path alone, until a crossing's landing was measured on each on the pair. Measured 2026-10-16 (mesh
f4ba8792; metal-microbench docs/measurement.md 2026-10-16, coded17-21 against lanes1-3, the pair's one-row split, 71
crossings a step): on the lanes the M5's FFN sum waits 14.4-16.8 us against 12.0-12.3 on its own queue pairs, the
Mini's stream lands about 3.5 us later (45.4-46.4 against 42.2-42.9 us), a step 5.28-5.34 ms against 5.10-5.21 (2-3 %);
a session's own start (nccl.h ncclMeshSessionWaits on the member that arrived last) 3.7-10.2 ms against 327-990 ms,
the exclusive queue pairs' making and pairing each session. At a session a call (A28), the start outweighs the steps'
cost below some 2,500 to 8,000 steps a call, so a direct program's flows ride the lanes; the exclusive form stays in
the code for a flow a READY asks it of, and goes with the next change that touches it.

A position is a whole slot on the wire (512 KiB of the library's large class; of its small one the communicator's
`small`, one block where none is given, the pair's decode 3,072 bytes, one frame), a filler
as much as a piece: TN3205 pairs a SEND with a receive of the same 4 KiB frames, and an exclusive ring posts its
receives ahead of what lands, so a position shorter than its slot would leave its receives' frames unmatched and every
later SEND of the ring a record off. A lane's header names the chunks and bytes of what follows it before its receives
are posted, so a lane can carry a position's own bytes (the producer naming them with its release, the receiver
posting that many frames and storing the words of the slot's records it skips at the last one's landing, a relay
carrying them on with its hand-off), and the library could then price a position at its bytes where its programs move
fillers or pieces short of a slot (at 16 MiB on six-node maps, up to twice a program's bytes). It is not built: the
exclusive form, the latency path every direct program of the pair takes, cannot carry it, so a price at a position's
bytes would be wrong on every exclusive flow, and the lanes' own path has not yet carried a crossing on hardware.
Once a crossing's landing is measured on each and the exclusive form is deleted, the lanes carry every position at
its own bytes and the library prices it so, one transport and one price (the library's request of 2026-10-15).

A port's lanes hold two queue pairs whatever its programs, routes and communicators. Over the review's seven
six-node cablings and 240 numberings each, one six-member communicator holds 4 queue pairs on the busiest device (its
lanes and its direct neighbour's exclusive flow; 6 to 18 at 112, refused on three cablings), and with four splits
beside it 8 (12 to 36 at 112, refused on all seven); no direct flow had to ride the lanes, and a fifth communicator
toward one neighbour would; the pair holds 4 where it held 2 (metal-microbench output_data/bridge-20261015
qp_lanes.py, qp_lanes.json; at the budget of 10 alike, qp_lanes2.py). The lanes' protocol, modelled with relays, stopped programs and reused table indices on
chains, rings, stars and a Studio K4, delivers every chunk to its record and completes every program that was not
stopped, 480 of 480 runs; posting the next header's receive before a header's records, posting no discards, or
reading no generation each misdeliver or desynchronise a lane in most runs (lanes_model.py, lanes_model.json).

## progress

A port's flows and lanes are served by the port's two progress threads, its receives' and its SENDs', started at its
first flow or its lanes and kept for the bridge's life: one pass serves every exclusive flow in the port's table (its
completion queues, its streams) and the port's lanes (their headers, their landings, their SENDs and completions) while
a flow in the table rides them or their SENDs or records are in flight, so a port of exclusive flows alone polls its
flows' queues and no other, as Q5P's threads did (mesh git 5702ea5, a thread a link and direction).
A flow enters the table when it is to be served and leaves it once neither thread reads it (an epoch each thread tells
at the end of the first pass that read it: the two threads read `high` and `epoch` on one line at every pass, and a
store of each pass's epoch by both bounced that line between their cores at the pass rate, a coherence miss in every
pass of the latency path; mesh git 765a8590^ told it every pass). MESH_VERSION 112 started two threads a flow, spinning while the flow lived, so a session kept 2(N-1)
spinning, and four a relayed pair: 10 to 34 on a member of a six-node mesh and 50 to 70 on a relay hub, above the
rank's encoding threads (QOS_CLASS_USER_INTERACTIVE). A node now runs two a paired port (at most two a cable,
independent of members, communicators and routes), and they spin only while something moves: after MESH_IDLE_NS
(100 ms) of nothing a thread parks on its port's `bell`, looking again every MESH_IDLE_POLL_NS (1 ms) while something
is its to look for (`port_pass`: a flow whose SENDs it serves, an exclusive flow's receives or a lane rider's headers,
a lane's records posted and not landed, a lane's SENDs in flight, a control entry queued), else on the bell alone, and
at once when the bell rings: a client's group of a program on the port starting (mesh-call.h `mesh_transfers_wake`), a
landing that released one of the port's SEND cells while its send thread was parked (`send_wake`: a relay's index
hand-off, the cell's `port` naming whose thread posts it), the bridge listing or unlisting a flow, queuing a control
entry, its lanes', stripe flow's or stream's change, its guard, or its stop. A thread woken by its bell with nothing to
look for parks again at once. Nothing lands on a lane for a flow this end has not listed (the peer sends only after
this end's POSTED, which follows the listing), so a node whose links are paired and which runs no program sleeps until
its bell rings, where the round's first pass looked every 1 ms for as long as its lanes were up (2,000 wakes a second
on the pair, 10,000 on a Studio of five cables). The ready
words a producer releases have no descriptor to wait on (RDMA-FIRST.md "Sleeps"), so a group's own ring is what ends
the park without a poll's delay; within a group, and for 100 ms after, the threads spin as before. A peer's landing on
a parked receive thread waits for its next look, at most 1 ms: the receive side's one event source is its completion
queues, which a completion channel (`ibv_req_notify_cq`) would wake at once, but that path has never run on this
provider and a destruction of a queue whose events are not acknowledged blocks (`ibv_destroy_cq`), so it waits for a
measurement. A send thread tells it is parked, then looks once more before it waits, and `send_wake` orders its
release before it reads that word, so one of the two sees the other; a thread reads the guard and the bridge's stop
again after it reads the bell's value it will wait on, so a guard raised or a stop between its loop's head and its wait
is taken, not slept past.

The bridge's waits on its progress threads are address waits on words they store, never a spin: `port_settle` rings
the bell and waits on each thread's `seen` word, which the thread stores and wakes as it passes the new epoch
(`port_told`); `port_guard` rings it and waits on each thread's `guarded` word, which a thread stores and wakes as it
holds; a held thread waits on the bell, telling the epoch as it moves, until the guard is lowered.

## control

A flow's lifecycle travels on RDMA, never on the control socket once it runs (the operator, 2026-10-17: "it's easy to
fix by remembering we're using rdma, not a hand-written sloplib breadboarding two cpus together"; metal-microbench
output_data/dead-history-20261017 SESSION-DESIGN.md, lifecycle-requirements.md R1-R9). The socket keeps what no queue
pair can carry yet: a link's bootstrap (HELLO, its lanes' and stripes' queue pair numbers) and a program's pairing
(READY, PAIRS, POSTED), and FAIL of a flow not yet running.

**The channel.** Each exclusive flow makes one queue pair past its data's, always: its control channel, its number
last in PAIRS (count one past the data's), a page of the window its own (`flow_make`: `CONTROL_SENDS` 16-byte slots its
SENDs read, then `CONTROL_RECEIVES` slots of 64 bytes its receives fill, posted at JOB_RECEIVE and each again as it
lands). A device without room for the pair, or a window without the page, makes the flow a lane rider. A frame is
`{kind, channel, count}`, 16 bytes:

| frame | sent | the receiver |
|---|---|---|
| ALIVE | the keepalive (`#keepalive`): at the configuration's interval while the flow runs, one in flight at a time | nothing: its receive is posted again |
| FAIL(node, code) | the flow failed (its program aborted, lost a peer or a link): its send thread posts it while the event loop asks it (FLOW_TELLING) | the flow fails (ECANCELED, ESTALE or ECONNRESET kept) naming the node, a relay's partner telling its peer on |
| RELEASED | a lane rider whose program ended in order was released (below) | a relay's flow passes it on and, once each end's came, releases both of the relay's flows; an end's flow takes nothing from it |

A lane rider's frames are entries of no chunks in its lanes' headers (lane 0: `ring` the kind, `bytes` a FAIL's node
and code), queued on its port (`port_tell`, the port's `tells`) and sent though the rider is gone.

**The orderly end** (`MESH_ORDERLY`, mesh-dataflow.c `mesh_detach`). A slot retired in order ends each running flow of
its program in order (FLOW_ENDING): its streams post every cell the program released (a stream is idle once its next
cell is not released: the client detached after its last publication, its GPU work finished), and its send thread
then tells the flow drained once those SENDs completed (`drained`, the event loop woken): an exclusive flow's once
every gate's requests equal its completions and no control frame is in flight (`flow_send`), a lane rider's once the
lanes' gates retired past the marks its last SENDs left on each lane (`lane_mark`, `lanes_drained`). The event loop
then releases it (`flow_released`), its program's words kept and its peer told nothing; a lane rider queues RELEASED
for the relays of its route. Its peer's flow is released when the peer's own client detaches, or fails as this end's
(ECONNRESET) at the READY of this end's next program of the execution (`#protocol`): nothing orders the two ends'
releases, since a client detaches only once every landing its program waited on has come, and a SEND's
completion is RDMA's own report that its data left. The client waits on none of it: its rows stay held by the bridge
(its slot's retired rows, released once no flow of it remains). A flow of an orderly-retired program that is not yet
running fails as an abort's does (FAIL on the control socket), since its peer's start cannot complete without it.

**Relays.** A relay's two flows carry a route's two directions, and no end's in-order release reaches them by another
event, so each end's lane rider tells RELEASED. A relay's flow that hears it passes it on by its partner toward the
route's other end (`relay_released`), and once each of its two flows has heard and passed one, both are released (their
lease and client slot then return): by then each end's program had every cell it waited on, all of which crossed the
relay, and had sent every cell it released. A relay of an end that fails fails with it (FAIL), or with a link's loss.

**The release** (R6). A flow is released by the event that ends it, its program's in-order end drained or a failure:
out of its port's table, then its queue pairs moved to the error state and destroyed (mesh-verbs.h `down_pair`: the
move ends the device's posted receives and SENDs, so no queue pair is destroyed ready to send with receives posted;
the move returns, and nothing waits on what it flushes), all under the port's guard (`port_guard`, mesh 542d0bc7: no
queue pair of the device is posted on meanwhile, design/RDMA-KERNEL-RECOVERY.md). A lane rider is released once its
records posted on the lanes have landed (the last landing wakes the event loop, FLOW_RELEASING: the peer sent each
header's data behind it, on the same lane) or the lanes that held them are gone. A destruction the driver refuses is
kept in the port's `remains` (`port_down`), tried again at each of the port's later teardowns and before its device
retires, and its flow is freed. Each release logs its census.

**FAIL.** A running exclusive flow's FAIL is told by its send thread (FLOW_TELLING, its program muted, FLOW_MUTED),
which posts it on the control queue pair and says so once its completion is taken (`said`); the event loop then stops
the flow, or sooner where the peer's own FAIL came or the link was lost (`port_lost`). Nothing waits on it on the
port's queue: a FAIL whose completion never comes holds that flow's queue pairs alone.

**The cancel** (R9, M12). A failed flow's program's words are cancelled past each ring's landed count alone
(`flow_cancel`): from its first record not landed, through the program's invocations less the ring's depth (a
program's readers keep within the depth of the landings, so those records' words hold landings already read; at least
the first, on which every reader waits before any later one), every record not landed in a one-pass program, after
the last landing the flow stores (its port's threads settled first). A word that landed stays, so a call whose
positions all landed completes. A GPU walk whose wait reads its word cancelled halts its later waits
(nccl-mesh-metal.m `mesh_step_check`), and the program's every flow was muted before any word was cancelled
(`program_quiet`), so a halted walk's later cells, computed without the lost peer's, are never posted: the peers fail
on its FAIL. Where the bridge itself exits, its clients cancel their own words (`#clients`, `mesh_watch`).

**Loss** (R7). By events alone: a link off (KEV_DL_LINK_OFF, KEV_DL_IF_DETACHED), the control socket's end or reset, a
peer's FAREWELL (its flows lost at once), a FAIL, an error completion, a client process's exit (EVFILT_PROC), a cancel or a
revoke (`mesh_retire`), and the map's prune (a program awaiting a node no paired path reaches fails, `#routes`). Nothing
declares a loss by silence: a peer whose program never runs its side, or whose host stops with its link up, is waited
for until one of these events, its client's cancel or revoke among them. On the loopback fabric's line of 4 (mesh-loop `kill`)
a client killed in a call fails each other member's call as a value within 1.1-2.5 ms, and a member whose own bridge
stops fails its call by its watch of the bridge (0.9 ms), its peers by the FAREWELL; a node's bridge whose binary is
replaced by rename under a call (mesh-loop `kill ... binary`, mesh 0fb8eb54) fails every member's call as a value, its
own member's in 2.6-3.0 ms and the others' in 1.0-2.3 ms, and starts again, the map converging, six times of six on
the host path and the walk (metal-microbench output_data/dead-history-20261017/scratch-delete/w1/run14).

**What still rides the control socket**, and whether the lanes could carry it: HELLO (a link's bootstrap: before any
queue pair, it cannot); READY with its program and route, PAIRS and POSTED (the pairing: an exclusive flow's queue
pairs are not connected yet, and a lane rider's index and generation are what PAIRS tells, so the lanes could carry a
lane rider's PAIRS and POSTED and an exclusive flow's READY, once the lanes are up); FAIL of a flow not yet running (as
READY); LANES and STRIPES (their own queue pairs' numbers: bootstrap); RECORD (the map, every node's record flooded: the
lanes could carry it, a header entry naming no flow, once up); FAREWELL (the lanes could, ahead of the socket's end; a
stopping bridge queues it on each socket and sends what the socket takes at once, never waiting on a peer that does not
read, and the socket's close follows); the requests' CHANNEL_* (services' byte streams, flow-controlled windows: the lanes could, as stripes do for
bytes); STREAM (its own queue pair's bootstrap).

## keepalive

A configuration's `keepalive_us` (etc/bridge/README.md; the bridge reads `keepalive`, nanoseconds) is the most an
exclusive flow's send side lets pass between frames on its control queue pair (`#control`) while the flow runs: an
ALIVE frame, one in flight at a time, posted once the interval passed since the last (`flow_control`), a paced emission
that nothing reads but the link, whose SENDs complete on the flow's SEND completion queue (`send_retire`, their wr_id
tagged `CONTROL_SENT_TAG`) and whose receives its receive thread posts again as each lands (`frame_heard`,
`CONTROL_HEARD_TAG`). A configuration that states none sends none. Each flow's census at its release is followed by its
control frames posted and heard and its keepalives. The frames ride the flow's own queue pairs, whose posts the flow's
quiescing stops before any of them is destroyed, as its data's. Its siblings are the configuration's `hello` (a port
with no peer says HELLO again at that interval, `#ports`) and `health` (the node's record flooded again at that
interval, `#ports`), each none where the configuration states none.

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

The lanes carried them first (mesh b21ced65, 9e0b3a4a) and must not: posting on the lanes' long-lived queue pairs while
a flow's were destroyed on the same device crashed the M5's bridge in `tbt_post_recv` (a store to an unmapped page at
a mapping's offset 8: design/RDMA-KERNEL-RECOVERY.md's September signature of the provider's shared doorbell mapping),
failed the next flow's first SENDs with IBV_WC_LOC_PROT_ERR, and left the lanes' frames unheard from the second call
on. Nothing posts on a queue pair that outlives a teardown of another.

## stripes

A client writes into a neighbour's window with no program at the neighbour: a stripe (the operator, 2026-10-01:
"MAKING K-MANY COMPUTERS INTO A SHARED ADDRESS SPACE WHERE EXTREMELY HIGH NUMBERS OF CPU CORES CAN ASYNCHRONOUSLY START
WRITING TO STRIPES OF THE REALLY BIG SHARED ADDRESS SPACE"; metal-microbench docs/acceptance-20261014.md A51: "The
bridge serves a neighbour's request over a link whether or not a local client has configured a matching program").
Any thread of any client, or a GPU, puts a request (the bytes at an offset of its node's window, the neighbour, a word
and a value) into a ring of its own that it registered with its bridge (mesh-call.h `mesh_requests_create`,
`mesh_write`); the bridge sends a descriptor and then the bytes, the neighbour's bridge posts its receives at the
same offset as the descriptor lands, and each bridge stores the value into its word as its side completes. A port's
stripes ride a flow of its own, two queue pairs made on demand and told the peer as the lanes are (STRIPES), served by
the port's progress threads (design/bridge-dataflow.md#stripes, which also holds the deviations from mesh 8fc27cbe,
where they were first built and from which they are restored, and the integration check). The descriptor's receive
posting is a header's, the one the bridge keeps for anything but the lanes' control: stripes are a client's writes
into memory, never a program's operands, which ride pre-posted exclusive flows alone (#lanes). A stripe flow that
failed running is made again as the next request toward its peer needs it; one that could not be made or connected
fails the next such request as a value by that code (its writer's word `UINT64_MAX`, its ring's code), and the request
after makes it again, so every attempt is a request's.

The flow's queue pairs live while the link is paired, so their posts would cross every other queue pair's teardown
on the device (#keepalive's hazard). While a port's device holds them, every change of the device's verbs objects
runs with the port's guard raised, the progress threads making no pass until it returns, and the stripe flow's own
teardown moves its queue pairs to the error state and then destroys them: the answer to the double
deregistration and the panics that ended 8fc27cbe (design/RDMA-KERNEL-RECOVERY.md "The stripe writes of October 1").
With no stripe flow on a port, its verbs work runs unguarded, as before.

## ledger

Where its configuration states `ledger = true` (etc/bridge/README.md), the bridge keeps a ledger of each flow: Q5P's
bridge trace (M27, the per-call bridge's MESH_LEDGER, mesh git 5702ea5, deleted with that bridge in 800fe6bc), which
located a crossing's stages on the pair, restored on the resident bridge, so that a crossing's ~20 us a hop today
can be set against Q5P's ~6 us bridge to bridge stage by stage (metal-microbench docs/measurement.md 2026-10-15 and
2026-10-16; Q5P is metal-microbench 533958a, mesh 5702ea5). The configuration is its one switch; the bridge's start
line says "ledgers kept".

- **What it holds.** A flow's ledger is made on its port's queue before either side is prepared or the flow listed
  (`ledger_open`). For each SEND cell its send thread first saw released (`flow_send` for an exclusive flow,
  `lane_send` for a lane rider): the cell, its stream's cycle, and when it was seen, when its first SEND was posted,
  when its last was, and when that SEND's completion was taken. A queue pair's completions come in the order posted,
  so the completion whose gate count reaches the cell's last SEND's is the cell's (an exclusive flow's; a lane's
  completions retire frames its flows share and are not attributed). For each receive record that landed
  (`flow_receive`, `lane_receive`): its index, its cycle, and when its word was stored. Every time is the bridge's
  `ledger_ns` (CLOCK_UPTIME_RAW, ns: CLOCK_MONOTONIC reads whole microseconds on macOS), each node its own.
- **Its bound.** A side's events are one pass of the flow's cells or records, and every pass of a cyclic program's,
  up to MESH_LEDGER_EVENTS (65,536) each; past it the flow runs on and the ledger keeps the first. They are allocated
  as the side is prepared, 56 bytes an event sent and 16 a landing (4.5 MiB a flow at the bound), with the cells' and
  records' names (32 and 16 bytes each).
- **Its cost.** Off: a branch on the flow's ledger pointer where a cell is seen, where its SENDs are posted, at a SEND
  completion and at a landing. On: a clock read and a store into the ledger's own buffers; nothing allocated, logged
  or locked on a progress thread.
- **How it is written.** At the flow's release (`flow_release`, once neither progress thread reads the flow and a lane
  rider's records have landed), on its port's queue: its header into the bridge's log, then a line an event, into the
  log where there are MESH_LEDGER_LINES (2,048) or fewer, else into a file beside the log,
  `ledger-<key>-<node>-<peer>-<stripe>-<ledger>-<UTC time>.jsonl`, which the header names (the file's first line is
  the header again). A flow held across executions (a program planned ahead, metal-microbench
  docs/placement.md#retention-and-protention-the-next-execution-planned-ahead) is written when it ends.

The lines, each a JSON object (in the log after its line stamp):

- the header, `{"ledger":id,"key":"<execution, hex>","node":n,"peer":p,"port":i,"near":a,"far":b,"stripe":s,
  "stripes":S,"exclusive":0|1,"queues":Q,"qps":R,"invocations":T,"cyclic":0|1,"started":ns,"sends":n,"send_room":N,
  "landings":m,"landing_room":M,"file":"<name>"|null}`: `node` this bridge's, `peer` the link's other node, `near`
  and `far` the program's two end nodes as this end names them (the flow at the link's other end names them the other
  way round), `id` the bridge's number of the ledger, which its event lines carry;
- a SEND cell, `{"s":[id,queue,invocation,position,binding,cycle,chunks,seen,first,posted,completed,bytes]}` (`bytes`
  the transfer's, -1 where the cell was not named; design/bridge-dataflow.md section 6);
- a landing, `{"r":[id,ring,invocation,position,binding,chunk,chunks,cycle,landed]}`.

A cell and its records share a name across the link: queue (the peer's ring of the same index, queue k's slot j), the
program's invocation, the position among that invocation's transfers of the queue (in their lists' order, which both
ends' streams and rings take, so the hardware pairs them), and the cycle; `binding` is the transfer's binding as its
list names it. A time not reached is 0: a lane rider's completion, a cell seen and not yet posted at the flow's end.

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
over its route's cables: its invocations t with t mod s = j are its stripe j of s (`stripe`, `stripes`), a flow on
each hop. A direct program's s is the pair's cables, flow j on cable j in the cable order both ends read alike
(`cable_rank`: the index the lower node gave each), so a bulk class's positions run over every cable at once
(MESH_VERSION 112 put a program's queue q on cable q mod n, so the large class rode one cable and a third carried
nothing). A routed program's s is the least common multiple of its hops' cables (`route_stripes`: a hop's the fewer of
its two ends' records', the first hop's this node's own), and stripe j rides cable (h + j) mod c of a hop of c cables
(`stripe_cable`, h the execution and the two ends mixed, alike at every bridge), so each cable of every hop carries s/c
of the program's invocations and a pair's routed programs spread over a hop's cables; past `MESH_STRIPES` (8) s is
the fewest cables of a hop, each stripe then a cable of its own on every hop, so the route still carries its slowest
hop's cables; on a relay a direction's stripes share one lease, each stripe's flows preparing its own
invocations of the lease's rows. At the round's first pass a routed or relayed program rode one cable a hop, priced
as all of them: over doubled Studio rings, chains and stars 4 of 18 to 20 of 20 routed pairs were priced at twice
what they could carry (metal-microbench output_data/bridgereview-20261015 routed_cables.py), and 12 of 20 on chains of
mixed counts (2, 3, 2, 3, 1 cables; 3, 4, 5, 2, 1), and over 2,000 random connected six-node cablings of 1 to 3
cables an edge 10,013 of 32,850 routed pairs, none now (routed_stripes_random.py); k routed programs over one doubled hop put 1.5, 2.25,
2.75 and 3.95 on its busiest cable for k of 2, 3, 4 and 6, now k/2, as priced (output_data/bridge-20261015
routed_stripes.py). A flow's receive ring posts its own invocations ahead by its share of
the program's span, so a stripe count that does not divide a cyclic program's invocations stripes it unevenly rather
than refusing it. Every READY carries its stripe, and a flow whose ends disagree (one end sees a cable the other does
not yet) fails as a value (EPROTO). The map prices a pair of several paired cables at their bandwidth together, and a
route at its slowest hop's (mesh-dataflow.c `mesh_paired`, `mesh_hop_cost`), which the stripes deliver where the
least common multiple is kept.

## routes

A routed pair's route is its lower node's: the lower end's program takes the route of its own map (mesh-dataflow.c
`mesh_route`, the library's routing table, mesh-plan.h `mesh_routes`, the same table every bridge and member makes of
the same records) and sends a READY a stripe along it; the higher end's program (`program_await`) takes at once the
stripes the lower's READYs left held, whatever its own map holds, and awaits the rest as they come, each take told the
await before its READY goes out, its status paired once every stripe runs. Where none is held yet and its own map holds
no route to the lower it is refused at once (EHOSTUNREACH naming the lower), as the lower is where its map holds none
to the higher; otherwise it awaits with no bound, ended by the lower's own failure as a value, its program's cancel or revoke (the
slot retired), or the map's prune of the lower (EHOSTUNREACH, `awaits_lost`). Each READY of its goes back along the
relays the lower's made, each mirroring the direction it carries and sending it on, so
every relay of the route carries both directions on one route (MESH_VERSION 112 routed each end on its own map: two
maps a link flap apart left both programs held on different relays, and a route of two or more relays carried the far
end's direction to its first relay alone). A READY that ends on a port where this node's program of its key toward the
sender runs flows elsewhere (one end saw a cable the other did not) is refused and the program failed at once, both as
values (EPROTO).

A relay's flows ride its ports' lanes, so a route costs the relays it crosses no queue pairs, and are released once
each end of the route has told RELEASED (`#control`), or fail with an end's failure or a link's loss. Each relayed direction
(an execution's key, its sending end and its receiving end: a pair of an execution's members whose route crosses the
node is two, one each way, and a pair of another execution, a split or shrink communicator of the same members, two
more; never one a node pair across executions) leases its rows and a client slot of the relay (`mirror`; one lease and
one slot whatever its stripes), its slots among the node's `MESH_CLIENTS` (64) with every slot its own processes hold
(libnccl-mesh's attach one a process and one a session), from the
window as a client's leases are (no reserve: the window is the clients' and the bridge's alike, and the planner counts
a relay's leases on every node a member route crosses, members or not, metal-microbench programs.py `sessions`; a
lease the window cannot give fails the relayed program as a value naming the relay): for each transfer the sender
sends, its slots of `mesh_relay_slot` bytes (its ring's depth less one slots of its invocation pages, and a last of
its bytes or its pages, the more), each rounded up to the window's block; the program's SEND cells and targets in one
lease (256 bytes a cell, a cell an invocation of each slot of each transfer and a terminal one, 16 bytes a target,
rounded up to the block); and the lists of its two destinations, a block each. At the library's session (two
classes, slots of 512 KiB and 16 KiB, ring depth 16, a cycle of 32 positions: nccl-mesh.c, `POSITIONS` 2 x `DEPTH`)
a relayed direction is 8,716,288 bytes (8.3 MiB), where it was 16,007,168 with the cycle of 8,192 positions and lists
of 4,096 entries (12,894,208 at that cycle with lists sized by use; output_data/bridge-20261015 footprint2.py); the
library states the same count for its own wire (nccl-mesh.c `relay_bytes`, ncclMeshRelayBytes), which a planner reads
in place of a figure of its own (11,813,376, the planner's of 2026-10-15, is the first pass's count at the cycle of 32
with lists of 4,096 entries). Beside the relays' leases a node's bridge holds its lanes' pages, 7 pages of 16 KiB a
port the region has (`lanes_reserve`: `MESH_PORTS` 8 of them, 917,504 bytes a node at its start, whatever its cables;
a port the window could not give them at the start takes them as its lanes are made), and its map's records.
