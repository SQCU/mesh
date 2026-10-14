# The bridge: a node's resident service

Each node runs one bridge, `rdma/mesh-flow`, as the launchd job `io.mesh.bridge` (install.sh), kept running by launchd
(KeepAlive) and checked by the keeper. Its lifetime is the node's, not any call's. No consumer starts, stops or
signals it. It takes no arguments and reads no environment: what it serves is found on the node, and what a program
needs of it is asked for by the program.

The bridge's work, and nothing else:

- it finds its ports and the node each one reaches, and keeps each link paired (`#ports`);
- it serves every client of its node at once, each with its own lease and programs (`#clients`);
- it pairs each program with the members' programs over the links it binds, runs its transfers and ends them
  (`#protocol`, `#flows`);
- it tells a program's failure to the program and to the peer, and pairs the link again for later programs
  (design/recovery.md).

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
block, the node's links (`MESH_PORTS`) and the client slots (`MESH_CLIENTS`). The arena is address space; a page costs
memory once something writes it, and it is registered with a port's device the first time a program names it on that
port (`device_register`, one 1 GiB extent at a time, never past the device's max_mr). Nothing in the region is sized by
a call: arena pages, page-table rows and transfer lists are a client's lease, taken as its program asks for them and
refused as a value where the node cannot give them.

A client slot (`struct mesh_client`) holds one client: its owner word (the client's generation, slot and process), its
execution's key, its program while it runs (`configured`), the bridge's hold on that program (`released`), its ring
depth and whether it is cyclic, and for each link the transfer lists' lengths, the program on that link
(`struct mesh_tx`: its SEND cells, its cancellation words, its transfer lists, the peer node it was bound for) and the
link's status for it (`struct mesh_port_info`: phase, code, domain, `prepared`). Every word a client writes is its own
slot's or its own rows'; no client reads, waits for or names another client's.

## ports

A port is an RDMA device of the node (`rdma_enN`, its interface `enN`). Every port is the bridge's from its start. On
its turn (every 3 s, design/recovery.md#pause), a port without a link opens its device (a context, a protection
domain), and while its port is active it sends a hello (its node and boot) to every node on its link (the link-local
all-nodes group, UDP `MESH_PORT`). Of two nodes that hear each other, the lower node dials the higher's listener on the
address it heard, and the two exchange their link hellos (node, boot, the port's LID and GID, the block size) over that
connection, its control socket, which the bridge keeps with a keepalive. The link is then paired: the region's link
table names the peer node, the phase `MESH_PAIRED`, the port's bandwidth and the pairing's index.

A node is its bridge's node number, a 31-bit fold of the host's UUID (`gethostuuid`). A cable moved to another port, a
cable added, a peer that restarted: each is a port that finds its peer again, with no configuration (there is none)
and no restart. Several cables between one pair of nodes are several ports, told apart by the port index.

## clients

A process attaches with `mesh_attach(context, name)`: it takes a vacant slot (or one whose process is gone, which it
retires) with one compare-exchange, and never waits for any other client. `name` is its execution's: every member of
one execution names it alike (an NCCL communicator's unique id, a call's name), and the bridges pair a program with the
members' programs of the same name. An attach is refused (EBUSY) when every slot is held, and ESRCH when the node's
bridge is gone.

A program is bound (`mesh_transfer_bind`: a link's lists are leased at its first transfer, its peer recorded),
prepared (`mesh_transfers_prepare`, `mesh_host_inputs` or the Metal transport) and started (`mesh_transfers_start`),
which returns once the bridge has paired the program on every link it binds, or with the failure the link's status
names. `mesh_transfers_stop` gives the program back; the bridge takes it off its ports and clears `released`.
`mesh_detach`, or the client's process exiting, retires the slot: its rows are closed, and the bridge releases them
once no port still holds a program of a client gone.

## protocol

Over a link's control socket the two bridges speak in fixed messages (`struct message`), each keyed by an execution's
key:

| message | sent when | the receiver |
|---|---|---|
| READY (queue pairs) | the program starts on this link | pairs at once where its own program of the key started, else holds the offer for the peer window |
| PAIRS (queue pair numbers, sequence numbers) | both programs started: its queue pairs made (INIT) | its queue pairs ready to receive (RTR), its receives posted |
| POSTED | its receives posted | its queue pairs ready to send (RTS), its progress threads started: the program runs |
| END | the program ended on this side | its program on the link fails (ECONNRESET, domain 4) |
| FAIL (code) | the program failed on this side | its program on the link fails (ECANCELED, domain 4) |
| FAREWELL | the bridge stops | the link's end is orderly |

No SEND of a program is posted before the peer's receives are (credit flow control holds a SEND until its receive is
posted, TN3205; POSTED orders the two). A program waits for its peer's READY at most the peer window (600 s) and for
its queue pairs to pair at most the pairing window (30 s), design/recovery.md#first; past either it fails (ETIMEDOUT).

## flows

A program on one port is a flow: its own queue pairs (as many as its highest queue on the link, times its slots, within
the device's max_qp), its own completion queues, its receive rings and SEND streams (design/prepared-machine.md M04-M30,
unchanged), and its own two progress threads, which run only while it runs. A link with no program running spins
nothing. A flow fails alone: its threads joined, its client's pending words on the link cancelled (M12), its status
`MESH_STOPPED` with the failure's code and domain, its queue pairs destroyed, its peer told. A failed flow is never
resumed; the next program on the link is a new flow.
