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
  them, and forwards the programs whose routes pass through its node (`#protocol`, `#flows`);
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
laid out from the node's memory (`mesh_layout`): an arena of every page of the node's memory, a page-table row a
block, the node's ports (`MESH_PORTS`), the client slots (`MESH_CLIENTS`) and each row's publication targets, one a
destination. The arena is address space; a page costs memory once something writes it, and it is registered with a
port's device the first time a program names it on that port (`device_register`, one 1 GiB extent at a time, never
past the device's max_mr). Nothing in the region is sized by a call: arena pages, page-table rows and transfer lists
are a lease, taken as a program asks for them and refused as a value where the node cannot give them.

The map is the bridge's table of records, one a node (`struct mesh_record`): its node number, name and boot, the
record's sequence in that boot, and its ports by index, each port's peer, phase, code, bandwidth and pairings. Its
`map_nodes` records lie at `map_off` in arena pages the bridge leases for itself and leases again larger as the mesh
grows (`mesh_records`), so no node count caps it. The map serial (`map`) is the bridge's to bump (`mesh_notify`)
before and after each change, odd while it writes; its own record is in before the region opens. A reader copies the
records and keeps its copy where the serial it read before was even and is unchanged after (`mesh_map_whole`), the
seqlock's reader (Boehm, MSPC 2012): mesh-plan.h `mesh_map`, with `mesh_map_wait` for the next map.

A client slot (`struct mesh_client`) holds one client: its owner word (the client's generation, slot and process), its
execution's key, its program once started (`configured`), its ring depth and whether it is cyclic, and its programs by
destination node, `MESH_DESTINATIONS` of them (`mesh_destination`). Each is the destination's node, its `struct
mesh_tx` (SEND cells, cancellation words, transfer lists of `MESH_QUEUES` queues each way, trace) and its status
(`struct mesh_port_info`: phase, code, domain, `failed` the node where it failed, the destination or a relay on its
route, and `prepared`). A client writes its own slot's other words and its own rows; the bridge writes the statuses,
the ports and the map. No client reads, waits for or names another client's.

## ports

A port is an RDMA device of the node (`rdma_enN`, its interface `enN`), given an index in the order the bridge finds
it: at its start, and again when an interface attaches or its link comes on that no port has (`ports_found`). On its
turn (every 3 s, design/recovery.md#pause), a port without a link opens its device (a context, a protection domain),
and while its port is active it sends a hello (its node and boot) to every node on its link (the link-local all-nodes
group, UDP `MESH_PORT`). Of two nodes that hear each other, the lower node dials the higher's listener on the address
it heard, and the two exchange their link hellos (node, boot, the port's LID and GID, the block size) over that
connection, its control socket, which the bridge keeps with a keepalive. A port whose device is retiring takes no
hello and no connection until its turn has retired the device and opened it again. The link is then paired: the
region's port table names the peer node, the phase `MESH_PAIRED`, the port's bandwidth and its pairings. The table
holds no device name or address: those stay in the bridge, and a reader names a port by its index (`mesh_observe`).

Each change to its port table is a new record of the bridge's node, its sequence one more, which the bridge sends as
RECORD over every paired port. A bridge that hears a record newer than the one its map holds for that node (a later
boot, or a later sequence of the same boot) puts it in its map and sends it on over its other paired ports, and a port
newly paired is sent every record the map holds: OSPF's flooding of link-state records (RFC 2328 §13). Every
bridge's map so holds every node's newest record, with no registry and nothing written by hand. A reader takes the map
from mesh-plan.h (`mesh_map`): two nodes are linked where each one's record holds a port paired with the other, and
members the links leave apart are joined along the map's route through other nodes. No reader joins records itself.

A node is its bridge's node number, a 31-bit fold of the host's UUID (`gethostuuid`), and its name the host's. A cable
moved to another port, a cable added, a peer that restarted: each is a port that finds its peer again, with no
configuration (there is none) and no restart, and its next record carries the change to every map. Several cables
between one pair of nodes are several ports, told apart by the port index.

## clients

A process attaches with `mesh_attach(context, name)`: it takes a vacant slot with one compare-exchange, never retires
another's and never waits for any other client; a slot whose process is gone is the bridge's to retire, on its turn.
Its rows and arena pages are claimed with a compare-exchange on each bitmap word that expects them clear (`mesh_scan`),
so two clients preparing at once never hold the same rows or pages. `name` is its execution's: every member of one
execution names it alike (an NCCL communicator's unique id, a call's name), and the bridges pair a program with the
members' programs of the same name. An attach is refused (EBUSY) when every slot is held, and ESRCH when the node's
bridge is gone.

A program is bound (`mesh_transfer_bind`: a transfer names its member node and queue, never a port; the destination's
lists are leased at its first transfer), prepared (`mesh_transfers_prepare`, `mesh_host_inputs` or the Metal transport)
and started (`mesh_transfers_start`). The start takes the bridge's map as it stands: the bridge pairs the program toward
each node it binds, over a port paired with that node or along the map's route through other nodes' bridges, and the
start returns once every destination is paired, or with the failure that destination's status names with its node
(EHOSTUNREACH where the map has no route to the node; ETIMEDOUT past the bridge's own windows, design/recovery.md#first;
the others mesh-call.h lists). A later loss fails the program alike, and the next start reads the map again.
`mesh_detach`, or the client's process exiting, retires the slot and ends its program: its rows are closed, and once no
port still holds a flow of the client the bridge releases them and the slot is vacant again.

## protocol

Over a link's control socket the two bridges speak in fixed messages (`struct message`), each keyed by an execution's
key:

| message | sent when | the receiver |
|---|---|---|
| READY (queue pairs, the route) | a program starts toward a node; the route is the nodes from the receiver to it | where the route ends at the receiver, pairs with its client's program of the key toward the sender, else holds the flow (`#flows`); where the route goes on, makes a relay flow and sends READY on with the rest of the route |
| PAIRS (queue pair numbers, sequence numbers) | both ends' flows made: its queue pairs made (INIT) | its queue pairs ready to receive (RTR), its receives posted |
| POSTED | its receives posted | its queue pairs ready to send (RTS), its progress threads started: the flow runs |
| END | the program ended on this side | its flow fails (ECONNRESET, domain 4); a relay's goes on |
| FAIL (code, node) | a flow failed, at the node named | its flow fails (ECANCELED, domain 4) naming that node; a relay's goes on |
| RECORD (a node's record) | the sender's port table changed, a newer record reached it, or the port paired | keeps it where it is newer than its map's, and sends it on (`#ports`) |
| FAREWELL | the bridge stops | the link's end is orderly |

No SEND of a program is posted before the peer's receives are (credit flow control holds a SEND until its receive is
posted, TN3205; POSTED orders the two). A program waits for its peer's READY at most the peer window (600 s) and for
its queue pairs to pair at most the pairing window (30 s), design/recovery.md#first; past either it fails (ETIMEDOUT).

## flows

A program toward one node is a flow on each port its route crosses: its own queue pairs (as many as its highest queue
toward the node, times its slots, within the device's max_qp), its own completion queues, its receive rings and SEND
streams (design/prepared-machine.md M04-M30), and its own two progress threads, which run only while it runs. A link
with no program running spins nothing. A client's flow runs over the client's own rows. On a node the route passes
through, the program is a relay's two flows, toward the route's previous node and its next, over rows the bridge leases
itself, the one's receive rows the other's SEND rows: a record landed from the previous node releases the SEND cell that
carries it on (design/prepared-machine.md M08), so a relay copies nothing and no process of its node takes part. On
the route's last node, while no client's program of the key has started there, the bridge holds the flow: it makes its
queue pairs and answers PAIRS at once, posts its receives (then POSTED) once that program starts and binds its rows, and
fails it toward the sender (ETIMEDOUT) where none starts within the peer window.

Each port's verbs work (opening, querying and retiring its device, making and destroying queue pairs, registering
memory) runs on that port's own thread, so a call blocked in the driver blocks that port alone, and the bridge's event
loop, its other ports and its clients' starts go on (design/recovery.md Q14).

A flow fails alone: its threads joined, its client's pending words toward the node cancelled (M12), its status
`MESH_STOPPED` with the failure's code and domain and the node where it failed, its queue pairs destroyed, its peer
told (FAIL), and on a relay the route's other flow failed with it. A failed flow is never resumed; the next program is a
new flow, on the map as it is then.
