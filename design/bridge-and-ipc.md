# The bridge: a node's resident service

Each node runs one bridge, `rdma/mesh-flow`, as the admin user's launchd agent `io.mesh.bridge`, which
`bin/mesh-user.sh CONFIGURATION [CHECKOUT]` writes and loads from the user's checkout (`~/mesh`), kept running by launchd
(KeepAlive) and checked by the keeper. Its lifetime is the node's, not any call's. No consumer starts, stops or
signals it. Its one operand is its configuration (etc/bridge/README.md), and it reads no environment: what it serves is
found on the node, and what a program needs of it is asked for by the program.

The bridge holds its node's RDMA for the node's clients. Its work, and nothing else:

- it registers its window, its configuration's share of the node's memory, whole on every RDMA device as the device
  comes up (`#region`);
- it finds its ports and the node each one reaches, keeps each link paired, and floods the mesh's map over the links
  (`#ports`, `#records`);
- on each link it makes the link's queue pairs once, as the link pairs, and leases them to programs, one program a
  lease (`#links`, `#leases`);
- it serves every client of its node at once, each a slot with its leases of the window and its program (`#clients`);
- it compiles each started program's transfers into chains of work requests on the queue pairs the program leased, and
  runs them on two threads a port that spin on the link's completion queues (`#programs`, `#loops`);
- it hosts the meeting at which a program's members agree on their leases, and forwards the member pairs whose routes
  cross its node (`#meet`, `#relays`);
- it carries requests, its clients' to the services of any node and its neighbours' to its own node's, admitting each
  against its node's memory, and serves node-info and, where configured, commands (`#requests`);
- it fails a program that loses a link, a peer or its client as a value naming the node, at the program and at its
  peers (`#programs`, design/recovery.md);
- where its configuration keeps them, it stamps each program path's crossings for its clients' ledger cuts (`#ledger`);
- a signal, or a change of its binary, stops it from any state with every resource released (`#stop`).

Its event loop waits on one kqueue with no timeout, every turn of it an event's: its sockets (the request listener,
the HELLO socket, the kernel's network events, each link's control socket, each request's), its clients' processes'
exits (EVFILT_PROC), its signals (EVFILT_SIGNAL), its own binary (EVFILT_VNODE), the power sources' and thermal
pressure's notify(3) descriptors, its configuration's paced emissions (EVFILT_TIMER: the `hello` and `health`
intervals, etc/bridge/README.md), and a user event its own threads and queues trigger (`wake`): a job's end on a port's
queue, a progress thread's word (`#loops`), the clients' control word (`control_watch`, an address wait on it), a
memory-pressure change (a dispatch source), a host name configd changed (an SCDynamicStore notification). Nothing on
the event loop reads a clock to decide; the send thread reads one for the keepalive's interval alone (`#keepalive`).

Operator, 2026-10-14 (D10): "clients are clients, bridge services are bridge services. clients shouldn't know anything
about other clients on the same mesh. ... clients *ask*, bridges *do*". And (D3), on whether the bridge keeps links
paired with no client attached: "this emerges from other requirements. required."

The bridge is updated by the mesh's own convergence (README.md Convergence; bin/mesh-converge.sh): the admin user's
agent `io.mesh.converge` builds the upstream revision in a stage, installs its images, and runs bin/mesh-user.sh on
them; a bridge whose binary changed stops at once, as a signal stops it (`#stop`), its programs and its peers' failing
as values, and launchd starts the new binary (KeepAlive). install.sh, the root layer, installs only `bin/mesh-stat`,
the reader node-info runs, built beside the admin user's running bridge.

## region

The node's one region is the shared memory object `/mesh0` (mesh.h `MESH_NAME`), made by the bridge at its start and
laid out by `mesh_layout`: an arena, the window, of the size its configuration states (the bridge's one operand, a
document its consumer writes in JSON, TOML or YAML, etc/bridge/README.md; no default; design/prepared-machine.md#M09,
with the operator's words: the window is the bridge's, its sizing for an execution the client's); a page-table row a
block (its page entry, the mapping and address of the block a lease bound to it; its buffer, the owner and the lease's
extent; its delivery, the address and per-invocation stride of the completion word a landing on it stores: 64 bytes,
and a bit in each of the row planes); the node's ports (`MESH_PORTS`, 8), with each port's lease words (`lease_off`,
`MESH_LEASES` a port, `#leases`); and the client slots (`MESH_CLIENTS`, 64). The region's tables are 64.5 bytes a
window page and 0.5 MiB besides (65 MiB beside a 16 GiB window, 57 MiB beside 14 GiB; design/prepared-machine.md#M09).

The arena is the registered window. Every port's device registers it whole when the device comes up, before any queue
pair exists (mesh-verbs.h `device_up`), in extents of 1 GiB (`MESH_REGION`, each inside one 4 GiB bank of the window's
bank-aligned base), and deregisters it only when the device retires (`down_device`), so no program's start or run
changes a device's mappings while DMA may be live (the operator's rule, quoted at M09). A device checks the window
against its own `max_mr` first; a window of more extents, or an extent the device refuses, is logged with the extent,
the errno and the extents held, and the port turns again at its next memory-pressure change or link event, keeping
the extents it took (design/recovery.md#registration). The registration reads the stop flag between extents (`#stop`).
The window is never clamped and never grows.

Nothing in the region is sized by a call: arena pages, page-table rows, transfer lists and publication targets are a
lease, taken as a program asks for them and refused as a value naming the execution where the window cannot give them.
A destination's transfer lists are leased at its first transfer with room for the entries one block holds (42 a list
at 16 KiB), and leased again twice as large, the entries copied and the old lease returned, when a list fills
(mesh-call.c `mesh_lists_grow`, at most `MESH_ORDERS` a list). A program's publications are its own:
`mesh_transfers_prepare` derives them from its SEND lists, one target a slot of each SEND transfer (the row it sends
and its column of SEND cells), and lays them after its SEND cells in their one lease, sorted by row, which the slot
names (`targets`, `target_count`); a producer's records (`mesh_publication_prepare`, design/prepared-machine.md#M10)
and a relay's landings (`#relays`) find a row's run by bisection (`mesh_row_targets`), and the slot's retirement
reclaims them with its other leases. The bridge leases from the window for itself as a client does: a page a port for
its keepalive frames, taken as the port is found (`#keepalive`), and the map's pages (below).

The map is the bridge's table of records, one a node (`struct mesh_record`): its node number and name, the record's
sequence (which orders a node's records, `#records`), its bridge's window in bytes, the node's memory in bytes and
hardware model (`hw.memsize`, `hw.model`), its ports by index, each port's peer, phase, code, bandwidth, pairings, the
queue pairs its device can use (`max_pairs`: its max_qp less one), and its link's queue pairs and leases (`pairs`,
`leases`, set at the link's pairing, which floods a new record, and 0 at its loss), and its health (mesh-plan.h `struct
mesh_health`, design/recovery.md#diagnosis): every node's window and model are in every bridge's map, read where the
records are (`#records`). Its `map_nodes` records lie at `map_off` in arena pages the bridge leases for itself and
leases again larger as the mesh grows (`mesh_records`), so no node count caps it. The map serial (`map`) is the
bridge's to bump (`mesh_notify`) before and after each change, odd while it writes; its own record is in before the
region opens. A reader copies the records and keeps its copy where the serial it read before was even and is unchanged
after (`mesh_map_whole`), the seqlock's reader (Boehm, MSPC 2012): mesh-plan.h `mesh_map`, with `mesh_map_serial` (the
serial now) and `mesh_map_wait` (the next whole map: an address wait on the serial, with no bound, until it differs). A
reader that meets the serial odd waits on it alike, its bridge's notification after the write or its exit ending the
wait. A record whose links (each port's peer, phase and bandwidth), name and facts are unchanged changes only its
health, its sequence and its ports' counts, under the health serial (`health`), so the map serial moves only when the
mesh does; `mesh_nodes_health` reads the records under both. A change the map does not take (a record that changes
nothing it holds, or one the prune drops at once) moves neither serial.

A bridge posts the notify(3) name `MESH_STARTED` (mesh-plan.h; a loop bridge's followed by its region's name) once its
sockets listen and its region is open, so a client that waits for its node's bridge blocks on that event
(`mesh_bridge_wait`) rather than trying again. The region's `bridge_pid` is the running bridge's process, 0 once it has
stopped (`#stop`).

A client slot (`struct mesh_client`) holds one client: its owner word (its generation, slot and process: generation <<
38 | slot << 32 | pid), its program once started (`configured`, the owner), its execution's key, its ring depth and
whether it is cyclic, its publication targets once prepared, and its program by destination node, `MESH_DESTINATIONS`
(64) of them (`mesh_destination`). Each is the destination's node, its `struct mesh_tx` (SEND cells, cancellation
words, transfer lists of `MESH_QUEUES` (4) queues each way, and the link it rides with each queue's lease on it, `port`
and `lease`, `#leases`) and its status (`struct mesh_port_info`: phase, code, domain, `failed` the node where it
failed, and `prepared`). A client writes its own slot's other words and its own rows, and claims and frees lease words
for its own slot alone (`#meet`); the bridge writes the statuses, the ports, the lease words otherwise, and the map. No
client reads, waits for or names another client's.

## ports

A port is an RDMA device of the node (`rdma_enN`, its interface `enN`), given an index in the order the bridge finds
it: at its start, and again when an interface attaches or its link comes on that no port has (`ports_found`); a device
the window cannot give a page for its keepalive frames is found again at the next such event. A port turns when an
event asks it (`wanted`): its finding, its link coming on or its interface attaching, a paired link of it lost (it says
HELLO again at once), the configuration's HELLO interval for a port with no peer (`hello`, a paced emission: a HELLO
lost on its link has no other event, and a port that never paired otherwise waits for its next event), or a
memory-pressure change after its window's registration was refused. On its turn, on its own dispatch queue, a port
without a link opens its device (a context, a protection domain and, once its port is active, the whole window
registered, `#region`; a device that fails otherwise is retired and opened again at once, and where that fails too, at
its next event), and while its port is active it sends a hello (its node) to every node on its link (the link-local
all-nodes group, UDP `MESH_PORT`). Of two nodes that hear each other, the lower node dials the higher's listener on the
address it heard (the higher answers each hello it hears, so the lower hears it at once), and the two exchange their
link hellos (node, the port's LID and GID, the block size; a peer of another block size is refused, EPROTO) over that
connection, its control socket. A port whose device is retiring takes no hello and no connection until its turn has
retired the device and opened it again. At the peer's hello the port's table in the region names the peer node, the
phase `MESH_PAIRED`, the port's bandwidth, the index the peer gave the cable and its pairings, and the bridge makes the
link's queue pairs (`#links`). The table holds no device name or address: those stay in the bridge, and a reader names
a port by its index (`mesh_observe`).

Each change to its port table is a new record of the bridge's node, its sequence one more, which the bridge sends as
RECORD over every paired port, as it does at an urgent change of its health and at the configuration's `health`
interval (a paced emission; none where it states none; design/recovery.md#diagnosis). A bridge that hears a record newer
than the newest it has heard of that node (`#records`) puts it in its map and sends it on over its other paired ports,
and a port newly paired is sent every record the map holds: OSPF's flooding of link-state records (RFC 2328 §13). Every
bridge's map so holds every node's newest record, with no registry and nothing written by hand. A reader takes the map
from mesh-plan.h (`mesh_map`): two nodes are linked where each one's record holds a port paired with the other, and
members the links leave apart are joined along the map's route through other nodes. No reader joins records itself.

A node is its bridge's node number, a 31-bit fold of the host's UUID (`gethostuuid`), and its name the host's. A cable
moved to another port, a cable added, a peer that restarted: each is a port that finds its peer again, with no
configuration (there is none) and no restart, and its next record carries the change to every map. Several cables
between one pair of nodes are several ports, told apart by the port index, each paired with its own queue pairs and
leases. The map joins the two nodes by one link at the bandwidth of the slowest of their paired ports (mesh-plan.h
`mesh_map`), and the bridges carry the pair's programs and channels over one cable, the one both ends take alike: the
lowest index the pair's lower node gave one of its cables to the other (mesh-flow.c `cable_to`, mesh-call.c
`agree_port_to`). Its other cables stay paired and carry records, and a program agreed after the first cable's loss
takes the next.

## links

A link is a port paired with its peer. At the peer's hello the bridge makes the link's queue pairs once (`link_make`):
n of them, the fewer of its device's usable queue pairs (`max_pairs`, its max_qp less one: RDMA-FIRST.md's "max_qp: 11
(10 usable)") and `LINK_PAIRS` (1 + `MESH_LEASES`, 17), each unreliable connected (UC, mesh-verbs.h `pair_make`) on the
link's two completion queues, its SENDs' and its receives', which every pair of the link shares (entries: n times the
device's frames a queue pair, at most its max_cqe; each pair's `share` of them its SEND gate and its receive ring's
capacity, `#loops`). Pair 0 is the keepalive's (`#keepalive`); lease l is pair l + 1 (`#leases`). It tells the peer every
pair in one PAIRS message (`pairs_tell`): each pair's index, number, first sequence number and round (the pair's
making), with the code and the SENDs completed that a lease's release carries (`#leases`). At the peer's first PAIRS
the link's count is the fewer of the two ends' makings, and the bridge connects each pair to the peer's of its index
(INIT, RTR, RTS: TN3205's one out-of-band step is the exchange of these numbers), posts the keepalive pair's receives,
sets every lease word below the count free (0) and the others `MESH_LEASE_DOWN`, tells the region the link's leases
(the count less one), starts the port's two threads once (`#loops`) and marks the link up, logging `link i (enN): node
p paired: n queue pairs, n-1 leases, f frames a pair`. A count under two loses the link (ENOSPC). A Thunderbolt 5
device gives 10: 9 leases a link, and the loopback fabric's devices as many.

The link's queue pairs are destroyed only at its loss, all of them together (`link_unmake`), and made at its next
pairing: no queue pair of a device is destroyed while another of it runs (`#loops`, RDMA-KERNEL-RECOVERY.md "October
16 and 17"). As a lease's holder lets go of it (`#leases`) its pair is made again in place (`pair_new`, mesh-verbs.h
`pair_reset`): moved to the error state (its posted receives and SENDs end, and nothing waits on what the move
flushes), to the reset state and to INIT, its number kept and its round one more, and the peer told its new first
sequence number and round (a PAIRS of that one pair); each end connects its pair to the peer's once the peer's told
round is its own (`pair_join`), so two ends that made one pair again in either order meet at the same round, and a
pair waits in INIT, taking nothing, until they do. A completion's work request id names its pair's index, the low 16
bits of its round and its record or gate position (`wr_id`: round << 48 | index << 40 | 40 bits), so a progress
thread skips a completion of a pair made again since, the flushed ones among them (`#loops`). Every making,
resetting, connecting and destroying is under the port's guard (`#loops`).

A link is lost (`port_lost`, design/recovery.md#detection) by events alone: its link off or its interface detached
(KEV_DL_LINK_OFF, KEV_DL_IF_DETACHED), its control socket's end or reset or a send the socket refused, the peer's
FAREWELL (`#stop`), a message of another magic or kind or a malformed PAIRS (EPROTO), an error completion on the
keepalive pair or a completion queue's poll that fails, and a verbs call of a pair's making, resetting or connecting
that fails, which also retires the port's device (closed, and opened again with the window registered on it before the
port pairs again: a device that refused every connect after the Mini's October 17 panic connected once a new process
opened it, RDMA-KERNEL-RECOVERY.md). The loss: the link down (`up` 0, its threads parked), every lease word `MESH_LEASE_DOWN` and the region's leases 0,
every program holding a lease of it failed as a value naming the peer (ECONNRESET where the peer said FAREWELL, else
the loss's code; `#programs`), the link's queue pairs and completion queues destroyed under the guard (`link_unmake`),
every request channel crossing it failed (`#requests`), and the port's phase `MESH_PAIRING`, so it says HELLO again at
once. Nothing declares a loss by silence: a peer whose program never runs its side, or whose host stops with its link
up, is waited for until one of these events, or its client's cancel or revoke.

## leases

A lease is one queue pair of a link, held by one program at a time. The region holds a word for each (`mesh_leases`,
`MESH_LEASES` a port): 0 free, a client slot's owner word while that slot claims or holds it, `MESH_LEASE_DOWN` while
the link is down or past its count. A program's slot names, for each destination, the port it rides (mesh_tx `port`,
the port index plus one) and each queue's lease on it (`lease`, the lease index plus one); its members choose them
together before its start (`#meet`), and the bridge checks them at the start (`#programs`). A queue of a program is a
lease: its SENDs and the receives its peer posts for them run in order on that one queue pair, and no other program's
work is on it.

The operator, 2026-10-16: "nobody gave permission for anything but a pre-posted exclusive flow". And 2026-10-18:
"it's called rdma. you have direct memory access ... why is there something you're calling a 'flow' on top of that
other than as suggestive metadata?" A lease is that exclusive, pre-posted crossing with nothing on top of it: its word
is its one state in the region, its pair's round its one state on the link.

A lease is let go when its holder's path ends (`path_release`), under the port's guard: the ring's words past its
landed count cancelled where the program failed (`#programs`), the pair made again in place (`#links`) and joined
where the peer's newer making was told, the peer told the new number with a code, and the word free. The code is what the peer's holder
does:

- **0**: the holder ended in order, or held nothing. The PAIRS carries `sent`, the SENDs the holder completed on the
  pair's last making. A program here holding the lease takes every landing of them first: once its ring's landed count
  reaches `sent` (its receive thread wakes the event loop at that count), its status toward that destination is
  stopped (`MESH_STOPPED`, ECONNRESET, naming the far node: its program there ended), its words kept, and it lets go of
  the lease in its turn with code 0 (`peer_released`); a relay's other side closes then (`#relays`). Where no program
  here holds the lease, the pair is made again and joined at once.
- **not 0**: the holder's program failed with that code (a lost link or peer, an abort, an error completion). A
  program here holding the lease fails as a value with that code, naming the node across the link (`#programs`), and
  lets go of its every lease with it, so a failure crosses a route lease by lease.

A lease claimed by a slot whose client is gone and held by no program is free again at the bridge's next turn
(`clients_turn`).

**Limits.** A Thunderbolt 5 link has 9 leases. A member pair takes one lease a queue it uses on every hop of its route
(`#meet`). libnccl-mesh's session binds every other member, two queues a member pair (rdma/NCCL.md), so on a line of
four members the middle link carries four member pairs, 8 of its 9 leases, and two such sessions hold it in turn, not
at once. Each member claims every lease free as it reports (`#meet`), so two sessions opening at one time on one cable
split its leases between their claims, and the later may find a hop with none (EBUSY). On the loopback fabric
(mesh-loop `share`, two communicators on one line, 32 calls each, three runs) sessions opened in turn shared the links
in every run; two opened at one time left the later communicator's first group EBUSY and its next group opened
(metal-microbench output_data/dead-history-20261017/scratch-rewrite/all.txt). A refusal is a value; nothing waits for
a lease.

## clients

A process attaches with `mesh_attach(context, name)`: it maps the region and takes a vacant slot with one
compare-exchange, never retires another's and never waits for any other client; a slot whose process is gone is the
bridge's to retire, on its turn. Its rows and arena pages are claimed with a compare-exchange on each bitmap word that
expects them clear (`mesh_scan`), so two clients preparing at once never hold the same rows or pages. A slot is one
execution's program: `name` is its execution's, every member of one execution names it alike (an NCCL communicator's
unique id, a call's name), and its key is the members' meeting's (`#meet`). A client that runs several programs takes a
slot for each on its one attach (`mesh_program_attach(program, client, name)`, sharing the client's mapping). An attach
is refused (EBUSY) when every slot is held, and ESRCH when the node's bridge is gone.

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
lists are leased at its first transfer), prepared (`mesh_transfers_prepare`, `mesh_host_inputs` or the Metal
transport, then `mesh_transfers_cyclic` where it runs without end), agreed (`mesh_transfers_agree`: its members'
leases and relays, `#meet`) and started (`mesh_transfers_start`, `#programs`). Its slot ends one of two ways:

- **In order** (`mesh_detach`): the owner word is left `MESH_ORDERLY` for the bridge, which closes each path of the
  program: every cell the program released is posted and its SEND completed, then each lease is let go with code 0 and
  the SENDs it completed (`#leases`); once none is held the bridge logs `program <key> (slot s) ended in order` and
  vacates the slot. `mesh_detach` writes one byte on each relay channel the agreement opened, which tells the relay
  that the route ends in order, and closes them (`#relays`).
- **An abort** (`mesh_retire`: a revoke, nccl.h `ncclMeshCommCancel`, or the client's process exiting) vacates the
  slot at once; the bridge fails the program at its next turn (ECANCELED), cancels its words (a revoking process's
  waits on them end there, a GPU walk's among them) and lets go of its every lease with that code, so its peers'
  programs fail too (`#leases`).

Either way its rows (a program's, or the leases of a lease-owning slot) are closed; once no program the bridge holds
belongs to a retired slot, the bridge releases them and returns their pages to the window in the same turn
(`mesh_retired_release`, then `mesh_reclaim`'s `mesh_buffer_reclaim`), so the window's unleased bytes (`mesh_window`,
node-info) count them free as soon as the client is gone; a lease that finds no room reclaims again before it fails,
as a backstop. The bridge watches each slot's process from the turn it first sees the slot (a kqueue `NOTE_EXIT` on its
pid, registered again for each new slot of the process, since the kernel keeps one watch per pid; a pid already gone
at that turn is retired then), and a process's exit retires every slot it holds, its lease slot and its programs'
slots alike.

A client watches its bridge alike (mesh-dataflow.c `mesh_watch`: a kqueue `NOTE_EXIT` on the bridge's pid, one thread
a process, over a mapping of the region it keeps, registered as the process first opens the region): at the bridge's
exit it marks the region's bridge gone (`bridge_pid` 0), cancels each of its own slots' programs' words (`UINT64_MAX`
where none landed, every one of a cyclic program's), fails each of their destinations' statuses (ESRCH) with its start
woken, moves the map's and the health's serials by 2 (their parity kept: a reader that found the bridge alive and had
not yet slept finds its word changed), and wakes the waiters on the map, health and control words, so no wait of the
process, a GPU program's on its landing words among them, outlives its bridge.

## programs

A program starts when its client sets its slot's `configured` (`mesh_transfers_start`) and the bridge's next turn takes
it (`program_take`). Its start is local: the bridge compiles it on this node from the program's lists and the leases
its slot names, and waits on no peer. For each destination with transfers it makes a path on the port the slot names
(`struct path`), and for each of the path's rings (queue k's slot s is ring k x slots + s, at most `MESH_QUEUES`; the
agreement allows one slot a queue):

- a SEND stream (`sends_configure`): every cell of the destination's SEND transfers, in invocation order and in their
  lists' order within one, each cell the chain of its chunks' work requests over its invocation's ring slot; a cyclic
  program's last cell followed by its first, a one-pass program's by its column's terminal cell, which no producer
  releases;
- a receive ring (`receives_configure`): the records of every RECEIVE transfer, invocation by invocation, in their
  lists' order within one, a record a chunk, each storing on its landing its slot's completion word (the row's
  delivery, among the program's cancellation words), or, on a row the program also sends on, that slot's SEND cell (a
  relay's, `#relays`), else a sink of its own; a record is posted once the record of its transfer a ring's span of
  invocations before it has landed (a slot short of the transfer's depth), within the ring's capacity.

Both ends derive the same order from the same lists (the members' programs of one execution bind each other's
transfers alike), so the hardware pairs each SEND with the receive posted for it: a SEND waits in the hardware for its
receive (TN3205's credit), and nothing else orders them. The operator, 2026-10-18: "you *do* know where the buffer is,
because the receive work request chose an address, and *you* wrote the code for the rdma users on both mesh clients."
A landing's word is the record's: the bridge stores the record's argument plus its cycle there
(design/prepared-machine.md#M30), and a reader waits on that word alone.

The start is refused as a value, each destination's status naming this node and the code, where a destination's port
is not paired (ENETDOWN), a list or lease is one the bridge cannot carry (EINVAL: a transfer of more slots than the
program's, a ring past `MESH_QUEUES`, a word outside the window, a queue with no lease or one past the link's count), or
a lease word is not the program's (ENETRESET: the link paired again since the agreement). Otherwise each path's rings
are attached to their leases under the port's guard (each record's work request id its pair's round and index,
`#links`), their first receives posted, and each destination's status `MESH_PAIRED` with `prepared` the owner, its
start woken. `mesh_transfers_start` returns 0, or the first refused destination's code; ESRCH where the bridge is
gone, ECANCELED where the program's own slot retired while it waited. From then the program runs on its ports' threads
alone (`#loops`): its producers release cells (mesh.h `struct mesh_send` `ready`, the cycle), the send thread posts
each released cell's chain on its lease as the lease's gate allows, and the receive thread stores each landing's word
and posts the ring's next records.

**Its end in order** (`MESH_ORDERLY`, `#clients`). The bridge marks each of its paths closing. The send thread marks a
closing path's ring drained once its stream is idle (its next cell not released) and every SEND posted on the lease
has completed, or the peer let go of it; the event loop then lets go of the path's leases with code 0 and the SENDs
completed (`#leases`), and once no path holds one the program has ended and its slot is vacant. Each peer's program
takes every landing of those SENDs, then stops its status toward this node (ECONNRESET naming it), its words kept:
libnccl-mesh reads that status when no position of its session is outstanding and opens the session again at its next
group (nccl-mesh.c `session_watch`). Nothing orders the two ends' releases, and nothing needs to: a client detaches
only once every landing its program waited on has come, and a SEND's completion is the hardware's report that its data
left.

**Its failure** (`program_fail`, design/algorithm-sources.md#meshresult). A program fails as a value naming a node: at
an abort of its slot (ECANCELED, this node), at an error completion or a refused post on one of its leases (that code,
this node), at a lease its peer let go of with a code (that code, the node across the link), at the loss of a link it
holds a lease of (ECONNRESET where the peer said FAREWELL, else the loss's code; the peer), and at the stop (ESHUTDOWN).
Each path's leases are let go with the code (`#leases`), so its peers fail alike. Its words past each ring's landed
count are cancelled, whether or not its client still holds its slot (an abort gives the slot up first, and the
program's rows stay its own until it is freed) (`ring_cancel`: `UINT64_MAX` from the first record not landed,
through the program's invocations less the ring's depth in a cyclic program, whose readers keep within the depth of
the landings, and every record not landed in a one-pass program; a landed word stays, so a call whose positions all
landed completes), the words of a destination never attached are cancelled whole (by the cancellation table its slot
named when the program was taken), and, where its slot is still its client's, each destination's status is
`MESH_STOPPED` with the code, the domain and the node. A GPU walk whose wait reads its word cancelled halts its later
waits (nccl-mesh-metal.m `mesh_step_check`), so a halted walk's later cells, computed without the lost peer's, are
never released. A failed program is freed once its slot retires. Where the bridge itself stops, its clients cancel
their own words (`#clients`, `mesh_watch`). Each failure is logged: `program <key> (slot s) failed at node n: <error>
(domain d)`.

On the loopback fabric (mesh-loop, a line of four and a torus of six bridges, metal-microbench
output_data/dead-history-20261017/scratch-rewrite/all.txt): libnccl-mesh's calls on the host path and the GPU walk ran
48 of 48 and 96 of 96; a communicator destroyed after its last group, 20 times on each path, ended every program and
relay in order (80 and 120 of each), none failed and no word cancelled; a member's client killed in a call, its
bridge stopped, and its bridge's binary replaced each failed every member's call as a value, the library's next calls
ran, and the replaced bridge came back.

## loops

The operator, 2026-10-18: "queues are an implementation detail of how the rdma driver works. the bridge does not
expose them. the queue is a data structure which should be spinqueried as FAST AS HARDWARE THREADS ALLOW on both sides
of each bridge, both to accept remote attempts to send, and to submit buffers which apis on the same side are trying
to send."

Each port has two progress threads (`port_progress`, QOS_CLASS_USER_INTERACTIVE), started once at its link's first
pairing and joined only at the stop: its receive thread (`mesh.rdma.receive`) and its send thread (`mesh.rdma.send`).
While the link is up and no guard stands, each passes without pause, reading the stop flag every pass:

- a receive pass (`receive_pass`) polls the link's receive completion queue: a keepalive frame's slot is posted again;
  a lease's landing stores its record's word (its argument plus the cycle, a release store), counts the ring's landed
  records, posts the ring's next records, and stamps the ledger where one is kept;
- a send pass (`send_pass`) polls the SEND completion queue, each completion retiring its pair's frames to its
  position; posts a keepalive frame where one is due (`#keepalive`); posts each held lease's released cells, request by
  request, while the lease's gate holds their frames (the frames posted and not retired, plus the next request's, at
  most the gate, the lease's share of the link's completion queues: `ibv_post_send` accepts past a queue's frames and
  corrupts later, so the gate is the bridge's); and marks a closing path's ring drained (`#programs`).

A completion of a pair made again since its post is skipped (`#links`). A thread's error (an error completion, a
refused post, a poll that fails) is a word on its pair (`failed`) and the event loop's wake; the event loop takes it at
its next turn (`pairs_turn`): a lease's is its holder's failure, the keepalive pair's or a completion queue's the link's
loss. The receive thread wakes the event loop too when a ring's landed count reaches the SENDs its peer said it
completed (`#leases`), and the send thread when a ring drains. The threads take no lock and allocate nothing.

While the link is down or a guard stands, a thread tells the guard value it holds at (`quiet`) and waits on the port's
`bell` (an address wait), which every change of either rings; at the stop it tells it holds no more (`UINT64_MAX`)
and returns. The guard (`guard_raise`, `guard_lower`) is the event loop's, raised around every change of the link's
verbs objects (a queue pair made, reset, connected or destroyed, the completion queues destroyed) and of what holds its pairs
(a path attached or let go): it makes the guard odd, rings the bell, and returns once each thread holds at that value,
so no queue pair of the device is posted on or polled while another changes. Posting on one queue pair of a device
while another of the same device was destroyed crashed a bridge in the provider's `tbt_post_recv` (a store to an
unmapped page at a mapping's offset 8: design/RDMA-KERNEL-RECOVERY.md's September signature of the provider's shared
doorbell mapping, and its "The stripe writes of October 1") and failed the next queue pair's first SENDs with
IBV_WC_LOC_PROT_ERR.

**Cost.** Two threads spin for every paired port, whether or not a program holds a lease on it: two cores a link. On
the loopback fabric every bridge of the mesh shares one host, so a line of four bridges spins twelve threads on an
18-core host; there a crossing's one-way hop read 26.0-26.3 us median (p10 14.3-14.6, p90 44.7-45.0) on a line of four
against 12.7-15.3 us (p10 5.5-7.0) on a line of two (mesh-loop `hop`, 3,000 rounds, two runs each,
metal-microbench output_data/dead-history-20261017/scratch-rewrite/hop.txt).

## meet

A program's members choose its leases together before its start, at one meeting (mesh-call.h
`mesh_transfers_agree(context, nodes, n, rank)`, after `mesh_transfers_prepare` and before `mesh_transfers_start`;
libnccl-mesh's session calls it as it opens). The meeting is the `meet` service of the first member's bridge
(nodes[0]), a request channel (`#requests`) like any other: each of a key's n members sends a `struct mesh_meet` (key,
n, its rank, its report's bytes) and its report and half closes; the bridge holds each until every rank of the key has
sent one, then answers each with them all (a uint32 n, n uint32 byte counts, the reports in rank order). A second
report of one key and rank ends the first's channel (ECONNRESET). n is at most `MESH_MEET_RANKS` (256) and a report at
most `MESH_MEET_BYTES` (1 MiB), else EPROTO. The bridge keeps nothing of a meeting once it has answered, and a channel
that ends (its member's interrupt or exit, a link of its route lost) leaves it.

The agreement (mesh-call.c):

1. Each member claims every free lease of each of its paired ports for its slot (a compare-exchange of each word from
   0 to its owner word) and reports its node, those ports (each its index, its peer, the index the peer gave the
   cable, its leases and the mask it claimed) and every destination its program binds (node, slots, invocations,
   depth, whether cyclic, and its transfers each way, queue by queue).
2. The first member plans on its own map (`agree_plan`): each pair of members that binds either way is routed
   (`mesh_route`, the routing table every bridge and member makes of the same records; EHOSTUNREACH where none), and on
   each hop of the route each queue the pair uses takes the lowest lease that both of the hop's member ends claimed,
   below the fewer of their counts, and that no other pair of the plan takes on that link (EBUSY where none; a hop
   between two relays is narrowed by no claim, and its relays' own claims decide, `#relays`). A pair with more than one
   slot a queue is refused (EINVAL). Each interior node of a route is given the pair's two directions to forward, the
   far ends' transfers and the leases on its two hops (`struct mesh_relay_entry`), all of one node's in one request to
   its `relay` service, whose answers the plan awaits; a relay's refusal is the plan's.
3. The members meet again (the key xor 0x6d656574206c6561), the first member's report the plan: its code and, for each
   member and destination, the port toward the route's first hop and each queue's lease. Each member writes its own
   into its slot (mesh_tx `port`, `lease`) and frees every claim the plan did not give it, all of them where the plan
   refused.

The agreement returns 0, or on every member alike the plan's refusal: EINVAL, EHOSTUNREACH, EBUSY or a relay's code;
ECONNRESET where a member's meeting was cut short, ECANCELED where the member's own slot retired. It waits on its
sockets alone, with no bound; `mesh_transfers_interrupt` from another thread ends the wait (libnccl-mesh's revoke
retires the session's slot, then interrupts it). The relay channels it opened are kept in the program's context until
`mesh_detach` (`#relays`). Every member of several meets, whether or not its program binds a destination; a lone
member (`n` 1) that binds none returns 0 at once.

## relays

A relay forwards a member pair whose route crosses its node, with no client of its own there: the agreement's first
member asks it (the bridge's `relay` service, mesh-plan.h `struct mesh_relay_request`: the key, the count of entries,
at most 64, and their bytes; each entry a `struct mesh_relay_entry` and its transfers, the far end's SENDs through
neighbour `a` toward `b` queue by queue, then the other way's). For each entry the bridge takes a client slot of its own
under the program's key (`mesh_slot_take`, among the node's `MESH_CLIENTS`) and mirrors the pair into it (`mirror`):
each transfer from `a` received into a section of its own (`mesh_relay_slot` bytes a slot, rounded up to the window's
block: its ring's slots but the last of its invocation pages, and the last its bytes or its pages, the more) and sent
from the same section toward `b` on the same queue and binding, and each transfer the other way likewise; the program
prepared as its ends' are, cyclic where they are; its port toward each neighbour the cable both ends take (`#ports`;
EHOSTUNREACH where none is paired); and each queue's lease toward each neighbour claimed, free or claimed by a live
slot of the same key not yet started (a member on this node that is also a relay of another pair: the leases it
reported free were planned for both), else EBUSY. It then starts as a member's program does (`#programs`), logs `relay
of <key> between nodes a and b (slot s)`, and answers 0 once every entry runs, its channel held open; else it answers
the refusal, every slot of the request retired and its claims freed.

A record landing on a relay's row stores into that slot's SEND cell toward the next hop (`receive_input`): the landing
releases the cell, and the send thread posts it on the next hop's lease, so a relayed row is sent on as it lands, with
no client and no copy. A relay's leases of the window are its route's to count (nccl.h `ncclMeshRelayBytes` states
them for libnccl-mesh's wire, which a planner reads: metal-microbench tools/mesh/programs.py `sessions`); a lease the
window cannot give refuses the request as a value.

A relay ends with its route. A neighbour that lets go of its lease in order (`#leases`) closes the relay's other side:
what landed from the first is sent on, the other side's leases are let go in order in their turn, and once neither
holds one the relay has ended in order (`program <key> (slot s, a relay) ended in order`) and its slot is retired. The
requester's byte past its request (written by `mesh_detach` where the program ended in order) says the route ends in
order; a relay channel that closes without it (the requester's abort or exit, a link of the channel's route lost)
fails the relay's programs (ECONNRESET), their leases let go with the code. A failure at either end crosses the relay
as a lease's code does (`#leases`). On the loopback fabric relayed calls ran 12 of 12 on a line of four and 30 of 30
on a torus of six, and every relay ended in order at each of the destroys above (`#programs`).

## requests

A node asks its own bridge, never a shell on another node (the operator, 2026-10-16: "why should a node do programming
by sshing into a terminal mode on a mesh peer instead of querying their own mesh bridge to ask 'hey what neighbors do i
have'?"). The bridge is the node's one daemon, as PRRTE's prted, Ray's raylet and the kubelet are each their node's.
Clients ask it about the mesh: `mesh_nodes_health` (`#records`) for every node's record, memory and health. They send
requests through it to the services of any node: `mesh_request`. ssh remains for administering the substrate alone
(installing or upgrading a bridge), as PRRTE uses it only to start its daemons.

A request is a channel (RFC 4254 section 5, the SSH connection protocol's channels; its `direct-streamlocal` channel
to a Unix socket on the far host is what `ssh node nc -U path` used). A client connects to its own bridge at
[::1]:MESH_PORT and sends a `struct mesh_request` (MESH_REQUEST_OPEN) naming the node, by its bridge number (its own
too), the service, and the bytes the request will take outside the window there (`#admission`). It reads the bridge's
int32 answer: 0 once the far end is open, else an errno.

- EHOSTUNREACH: the map holds no path to the node.
- ENOENT: the node serves no such service.
- ECONNREFUSED: the service's socket refused.
- ENOMEM: the node's available memory is short of the request's bytes (`#admission`).
- ENOSPC: the bridge holds `MESH_CHANNELS` (256) channels already.
- the code of a link of its route lost before the far end answered (ENETRESET where it names none).

A channel unanswered waits for its answer, its route's loss or its client's end, with no bound. The connection then
carries the client's bytes to the service and the service's back. Each direction ends with its writer's half close
(shutdown SHUT_WR), and a channel that fails anywhere resets the connection.

A channel's messages ride the links' control sockets beside the bridges' own (`#wire`):

| message | the receiver |
|---|---|
| CHANNEL_OPEN (its key, origin and destination; the service; the sender's window; the bytes) | at the destination, admits it and connects the service's socket (or opens its own service) and answers CONFIRM, or REFUSE with why; elsewhere, opens the next hop along its own map's route and answers once that hop does |
| CHANNEL_CONFIRM, CHANNEL_REFUSE | answers the hop the OPEN came from, and at the origin the client |
| CHANNEL_DATA | gives the bytes to the channel's other end |
| CHANNEL_WINDOW (bytes) | may send that many more bytes toward the sender |
| CHANNEL_SHUT | that direction ended: the other end is told once its queue is given |
| CHANNEL_CLOSE (code) | the channel failed: every other end is told (a socket reset, the next hop CLOSE) |

- **Keys.** A channel's key is its origin's bridge number and sequence, the same on every hop. A bridge between holds
  the channel as two ends, each a port, and needs no route of its own beyond the next hop.
- **Windows.** Each hop grants its sender a window (128 KiB), and grants bytes back only as they leave toward the other
  end (RFC 4254 section 5.2), so a hop holds at most a window each way. A socket end is read only while the other end
  takes at once what is read.
- **Losses.** A link lost fails every channel crossing it, and each channel's other ends are told.
- **Registration.** A service registers its listening Unix socket (MESH_REQUEST_SERVE: its name and its socket's path)
  on a connection it holds open, at most `MESH_SERVICES` (32) at once. The registration lasts while that connection
  does, and a newer one of the same name replaces it. The resident rank registers `rank` (metal-microbench rank.swift),
  and registers again at its bridge's start (`mesh_bridge_wait`).
- **The bridge's own services.** `meet` (`#meet`), `relay` (`#relays`), `exec` (`#exec`) and `nodeinfo`: a channel
  to `nodeinfo` is a connection to the node's own node-info (io.mesh.nodeinfo, [::1]:8100, else 8099), so a driver
  reads every node's node-info by its bridge number. It needs no address and checks no identity.
- **Ledger cuts.** A request of kind MESH_REQUEST_LEDGER names a program's key and a file, and is answered at once
  (`#ledger`).

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
(`#clients`). A request that names none, a cancel's or a meeting's, is admitted at any pressure. The health record
carries `available` beside `free` (the kernel's free pages, which leave out the cache it reclaims): on the Mini with
its rank resident, 0.27 GiB free against 5.3 GiB available (2026-10-16).

The bridge also watches its node. While the kernel's memory pressure is at warning or worse, or the available memory
is under a sixteenth of the node's (1.5 GiB on the Mini, 8 GiB on the M5), it logs the largest processes its user
runs, by physical footprint (`memory low: ...; largest: <name> (<pid>) <GiB>, ...`). It logs this as the condition
begins and again as the kernel's pressure changes while it lasts. That names what holds memory outside the window,
whether the holder is a bridge client or not; the bridge never ends a process for it.

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

A configuration that does not state `exec` true refuses the service (EACCES).

## records

A node's records are ordered by their sequence, and of two of one sequence the one of greater bytes is the newer (RFC
2328 §13.1's checksum, `record_newer`). Each bridge keeps, apart from the map, the newest record it ever heard of each
node (RFC 2328 §13's database, mesh-flow.c `struct known`), and takes and sends on a record only where it is newer than
that, the map holding the node or not. The map's prune (a node no paired path reaches leaves it, §14) drops a node's
record but not its entry, so a copy of a record the map dropped, still in flight on a cycle of survivors after its node
stopped, is not taken as fresh and sent on again: it ends at the first bridge that has heard it. A record older than
the newest heard is answered by that newest, sent back on the port it came from (§13 step 8); one equal to it, of a
node the map does not hold, is put again only where a paired path now reaches its node (the paths came back before
its node's next record), and sent on only then. A bridge's own sequence starts at 1 each time it starts; a record of
its own node newer than the one it holds (an earlier start's, flooded back to it as a link pairs, or answered by step
8) moves its sequence past it and floods its record again (§13.4), so a node whose bridge started again is heard again
by sequence alone. A full table of 256 entries gives its place to the first entry whose node the map does not hold.

A record carries its node's facts with its links: its bridge's window (the bytes its configuration registers), its
memory and its hardware model. They are the driver's to read from its own bridge's map, every node's at once, rather
than each node's node-info over ssh: the library's reader copies them (mesh-plan.h `struct mesh_node_health`,
`mesh_nodes_health`) and mesh-stat prints them in each of its `nodes` (`window`, `memory`, `model`).

## ledger

Where its configuration states `ledger = true` (etc/bridge/README.md), the bridge keeps a ledger of each path of each
program (its side toward one destination): Q5P's bridge trace (M27, the per-call bridge's MESH_LEDGER, mesh git
5702ea5), which located a crossing's stages on the pair, kept on the resident bridge, so that a crossing's hop can be
set against Q5P's ~6 us bridge to bridge stage by stage (metal-microbench docs/measurement.md 2026-10-15 and
2026-10-16; Q5P is metal-microbench 533958a, mesh 5702ea5). The configuration is its one switch; the bridge's start
line says "ledgers kept".

- **What it holds.** For each SEND cell of the path, as its send thread posts it (`stream_post`): the cell, its
  stream's cycle, when the thread first saw it released, when it posted its first SEND (after the lease's gate) and
  when its last. For each record that landed (`receive_pass`): the record, its cycle, and when its word was stored.
  Every time is the bridge's CLOCK_UPTIME_RAW in ns (CLOCK_MONOTONIC reads whole microseconds on macOS), each node its
  own. No completion is stamped.
- **Its bound.** 65,536 events of each kind a path (`MESH_LEDGER_EVENTS`), each a ring the newest overwrite: 40 bytes
  a SEND event and 24 a landing, 4 MiB a path outside the window, allocated as the program is taken, with the names of
  the path's cells (40 bytes a cell).
- **Its cost.** Off: a branch on the path's ledger pointer where a cell is seen and posted and at a landing. On: a
  clock read and a store into the ledger's own buffers; nothing allocated, written or locked on a progress thread.
- **How it is read.** A client cuts it (MESH_REQUEST_LEDGER: mesh-plan.h `mesh_ledger(name, path)`, nccl.h
  `ncclMeshLedger(comm, path)`): the bridge appends to the file `path` (absolute, under 104 bytes), on its event loop,
  for each path of each running program of the key that keeps a ledger, one header line and then the path's events
  since the previous cut (the newest 65,536 where more came). It answers 0, ENOENT where no running program of the key
  keeps one, ENOTSUP where the configuration keeps none, EINVAL for a path not absolute. The bridge's log holds no
  ledger, and the bridge writes no ledger file but at a cut.

The lines, each a JSON object:

- the header, `{"ledger":id,"key":"<execution, hex>","node":n,"peer":p,"port":i,"near":a,"far":b,"invocations":T,
  "cyclic":0|1,"started":ns,"sends":S,"landings":L,"cut":[s,l]}`: `node` this bridge's, `peer` the node across the
  link, `port` its index, `near` this node and `far` the path's destination (a member's far member, a relay's
  neighbour), `started` when the program was taken, `sends` and `landings` the path's events so far, `cut` the counts
  at the previous cut, and `id` the bridge's number of the ledger, which its event lines carry;
- a SEND cell, `{"s":[id,ring,invocation,position,binding,cycle,chunks,seen,first,posted,0,bytes]}` (`posted` its last
  SEND's; `bytes` the transfer's, -1 where the cell was not named);
- a landing, `{"r":[id,ring,invocation,position,binding,chunk,chunks,cycle,landed]}`.

A cell and its records share a name across the link: ring (the peer's ring of the same index), the program's
invocation, the position among that invocation's transfers of the ring (in their lists' order, which both ends' streams
and rings take, so the hardware pairs them), and the cycle; `binding` is the transfer's binding as its list names it.

`rdma/ledger.py PATH...` reads two nodes' cuts of an execution, pairs each path with its peer's across the link (the
same key, the nodes and `near` and `far` each the other's) and a cell sent with its last chunk landed, and gives each
direction's seen to first post (the send thread's pass and gate), first to last post, and the one-way bridge-to-bridge
hop with the clocks' offset cancelled: each crossing paired with one the other way, of the same name where the program
crosses back there, else the nearest in time, hop = ((landed_B - seen_A) + (landed_A - seen_B)) / 2, as Q5P's analysis
took it (an offset so taken holds the two directions' hops alike). It pairs the paths of a link whose two nodes are the
paths' two ends; a routed pair's hop to its relay names a far member at one end and a neighbour at the other, and is
not paired. Not kept: Q5P's third time of a landing (after its ring's next records were posted) and its layout lines
(`trace_layout`, `trace_binding`), whose work the names do.

## keepalive

A configuration's `keepalive_us` (etc/bridge/README.md; the bridge reads `keepalive`, nanoseconds) is the most a link
lets pass between frames on its keepalive queue pair (pair 0, `#links`) while a program holds one of its leases: the
send thread posts a 64-byte KEEPALIVE frame (a `struct message`, from the port's page of the window) once the interval
has passed since its last, while fewer than 16 are in flight; the peer's receive thread posts each of its 16 receive
slots again as a frame lands (`receive_pass`), and nothing else reads them. A link no program holds a lease on carries
none, and a configuration that states none sends none. The frames ride a queue pair no program holds, made once at the
link's pairing, so no lease's release touches it. Its siblings are the configuration's `hello` (a port with no peer
says HELLO again at that interval, `#ports`) and `health` (the node's record flooded again at that interval,
`#records`), each none where the configuration states none.

Why: on the pair (metal-microbench output_data/split-20261017, 2026-10-17) the Mac mini's Thunderbolt RDMA path, quiet
for the 40-50 us of a layer's window, completed its next SEND 6.2 us after its post where the M5's took 2.5, and
signalled a frame the M5 sent while it computed up to 10.6 us later; with a frame of the peer's landed 10 us before, its
SEND completed in 2.7 us (completion.py, contention.py: each crossing's two one-way times by how far apart the members
published). The provider's completions are a ring the controller writes and the poll reads in user space
(`tbt_poll_qp_send`, `tbt_poll_qp_recv`: a descriptor's bit 0x15), so the time is the controller's, not a parked
thread's: the send thread saw each released cell within 0.04 us. A host-path exchange of 3,072 bytes read 12.8-13.8 us
a round after gaps of 0-10 us, 15.1-15.6 after 20 and 16.5-18.8 after 40-160 (gapprobe2); UDP on the same cable did
not keep it awake (gapprobe3). With frames every 5 us on the link, through the one call in which they flowed, the
Mini's SEND completion read 1.8-1.9 us and the bridge-to-bridge hop 2.98 us median, 2.60 p10, 4.21 p90 against 6.50,
5.08 and 8.88 (ka3 against tc62), the step's median 3,904 us, p10 3,673, mean of the last 120 4,029 (mesh 12f484f8).

## stop

The operator, 2026-10-18: "imagine a number for every single state the bridge can be in. for all numbers there must be
a signal which can be sent to the bridge which causes the bridge to shut down as an application and release all
resources."

The bridge has one stop flag (`stop`, mesh-verbs.h), set by a signal's handler (SIGTERM, SIGINT and SIGHUP: `stopped`,
which keeps the signal as the reason) and by the event loop at its binary's change (EVFILT_VNODE: the file it started
from deleted, written, extended, renamed or revoked; the reason `MESH_STOP_BINARY`); the kqueue's EVFILT_SIGNAL wakes
the event loop. Every loop reads it every pass: the event loop's turn, each progress thread's pass (a parked one is
woken by its bell), the clients' control watcher, and a device's registration between extents (ECANCELED). No thread of
the bridge waits on a peer, a client or a drain: a progress thread waits on its bell alone, the guard on the progress
threads alone, which hold at their next pass or have returned, and the event loop on its kqueue alone.

The stop, in order: `bridge node n: stopping (<reason>)`; each port's bell rung and its two threads joined; every
request channel ended (ESHUTDOWN) and every request socket closed; each port told FAREWELL with the reason, as far as
its socket takes it at once, and lost (ESHUTDOWN, its device retiring): every program holding a lease of it failed as a
value, its queue pairs moved to the error state and destroyed, its completion queues destroyed (`#links`); every
program freed; the region's `bridge_pid` 0 and the clients' control word notified, so every client's watch and wait
ends (`#clients`); the control watcher joined; each port's device closed on its own queue (its window's extents
deregistered, its protection domain and context closed, in design/RDMA-KERNEL-RECOVERY.md's order), behind a
registration under way, which returns at the flag; `bridge node n: stopped: q queue pairs destroyed, p ports' devices
closed`; the region unmapped and unlinked; exit 0. A peer that hears the FAREWELL loses the link at once (ECONNRESET,
its programs failing as values naming this node), and logs the reason (design/recovery.md#diagnosis).

On the loopback fabric (mesh-loop `sigterm`), SIGTERM to every bridge of a line of four (three rounds) and of a torus of
six, in each state the loop makes (idle, pairing, a host-path call, a GPU-walk call, a relayed call, after a link loss,
with a client killed, with a peer's bridge stopped): every bridge exited 0 with its stopped line, no region and no queue
pair left, in 1-6 ms (metal-microbench output_data/dead-history-20261017/scratch-rewrite/all.txt). SIGTERM is
launchd's stop; a bridge never receives SIGKILL from the mesh (RDMA-RULES.md).

## wire

The bridge's whole wire vocabulary is one enum (mesh-flow.c `enum wire`):

    enum wire { HELLO=1, PAIRS, FAREWELL, RECORD, CHANNEL_OPEN, CHANNEL_CONFIRM, CHANNEL_REFUSE, CHANNEL_DATA, CHANNEL_WINDOW,
                CHANNEL_SHUT, CHANNEL_CLOSE, KEEPALIVE };

| member | carried on | sent | its receiver |
|---|---|---|---|
| HELLO | UDP to the link-local all-nodes group, then the control socket | by a port whose device is open: its node, LID, GID, block size | the lower node dials the higher; on the socket, the link's greeting (`#ports`) |
| PAIRS | the control socket | at a link's making, and at each lease's making again, with its holder's code and SENDs completed | pairs joined; a lease's holder fails, or takes its last landings and lets go (`#links`, `#leases`) |
| FAREWELL | the control socket | by a stopping bridge, with its reason | the link lost at once (`#stop`) |
| RECORD | the control socket | a node's record, flooded | the map (`#records`) |
| CHANNEL_* | the control socket | a request's channel, hop by hop | `#requests` |
| KEEPALIVE | the keepalive queue pair | by the send thread at the configured interval while a program holds a lease of the link | nothing but its receive posted again (`#keepalive`) |

Each is a `struct message`: its magic (0x4d595048 plus `MESH_VERSION`), its kind, the sender's node, block size and
port, its LID and GID, a code, a key, a count and its payload's bytes (at most 64 MiB). A message of another magic or
of a kind outside the vocabulary loses the link (EPROTO). Every program datum crosses as SENDs on its leases. No
message describes data to a reader, and none orders, gates, acknowledges or schedules a crossing: a landing's word is
the only signal a reader acts on. A PAIRS's `sent` tells the peer's bridge alone how many landings of a lease to take
before it lets go of the lease (`#leases`).

Adding a member to this enum, or a state to the bridge (a lease word's value, a program's state, a link's phase, a
message a loop waits for), is a design change. It needs an operator's quote, verbatim and dated, in this section before
it is written. What a reviewer holds it to is the record and the rule of 2026-09-08 and 09-09, restored here verbatim.

From d8d69cca:design/distributed-reduce.md:203-242 (2026-09-09):

### What was done instead, and why it is forbidden

Every divergence from this document that reached the repository was a
control-flow object standing in for something the page table already held,
written by an agent that had not read the prior art and did not want to. The
record is kept here so that the next such object is recognized on sight.

- A `K_DIGEST` control frame carrying the sent hash, a mismatch poisoning the
  whole program through the status word, and a paragraph added to this
  specification to bless it (2026-09-08, `f47fc4e`, `2188f44`), after the
  operator had said that acknowledgements are a private feature of the
  transport and not the algorithm. The digest is a page.
- A `K_OPEN`/`K_READY` rendezvous with exponential backoff that held every
  page until a "ready" message arrived, and `K_ABORT`/`K_ABORTED` propagation
  between peers: a hand-rolled connection protocol on a link that already
  has flow control. Its one real function — not sending to a peer whose client
  had not attached — was the receiving bridge dropping frames addressed to it;
  the bridge now holds them.
- A runtime reduce node that counted generations by one where a row family
  advances by the number in flight, so families past the first never reduced;
  it was committed as measured without a run that could have shown it.
- Hidden-state rows read as a contiguous matrix across two pages, crossing the
  page headers; the paged path was committed with a relative error copied from
  the CPU path it replaced.
- The digest page used as the gate on the next stage's issue — the numerical
  check made a dependency of the numerical program, which this document
  forbids in two places.
- A driver with phases, per-job state, per-call tasks awaiting predictions,
  completion words, and fixed 128-row groups: a scheduler. It issued about two
  thousand GPU command buffers per evaluation and made the pass three times
  slower than the barrier it was meant to remove. The readiness it tracked was
  already in the stamps.

None of these was hard to avoid. The token store, the in-data flag, the
region runtime, the timestamped epoch and the row-streamed reduce-scatter are
each thirty to fifty years old, documented, and measured; reinventing a piece
of one from a message-passing reflex is not engineering caution, it is not
having done the reading. A change to this repository that adds a frame kind,
a handshake, a retry, a phase, a token, a gate, or a scheduler is wrong before
it is measured, and measurement has so far agreed every time.

From 1347b7aa:design/pages-and-functions.md:156-212 (2026-09-08):

### Addendum: waiting for messages about data, on infinitely many Minis

Suppose a Mini does not wait on its input rows. Suppose it waits for a
message that says the data is coming, or is done, or may be used: a token, a
gate value, a "ready" flag, a scheduler's decision, a completion callback that
stands in for the pages. Then it acts on the message. This is what goes wrong.

1. **The message is a second thing on the same link.** The pages take one
   hop. The message about them takes a hop too. Every dependency on the chain
   now costs a hop for the pages and a hop for the message. On a chain of
   depth D the NFE waits for D messages it did not need. On an infinite mesh
   D is not bounded, so the extra wait is not bounded either. Waiting on the
   rows costs nothing extra: the row is stamped when the page lands.
2. **The message can be true and the data still absent.** A message says
   something about the sender's state. The pages may still be in flight, or
   lost, or land out of order, or be corrupt. Acting on the message is acting
   on a claim. To make the claim safe, the sender and receiver need
   acknowledgements, retries, and ordering. That is a protocol on top of page
   delivery. The digest, computed after the pages were used, already finds a
   wrong page without any of it.
3. **The message can be false and the data present.** The pages landed and
   the rows are stamped, but the message is late, or the sender concluded
   early, or the link that carried the message broke while the link that
   carried the pages did not. The Mini sits with everything it needs and does
   nothing. Waiting on the rows would have run the function at once.
4. **The wait binds this Mini to another Mini's control flow.** A message
   comes from a sender's scheduler, not from data. If that scheduler is slow,
   this Mini is slow, and every Mini downstream is slow, even though their
   pages have all arrived. On an infinite mesh there is always a slowest
   scheduler somewhere, and the chain of "waiting for whose message" has no
   end. Waiting on rows binds a Mini only to the pages in front of it.
5. **Work handed to the GPU while it waits for a message is killed or idles.**
   We measured this: a command buffer that waits on a signal for seconds is
   ended by the watchdog, and every buffer behind it dies with it. A GPU given
   only functions whose rows are stamped runs at its roofline fraction. Its
   utilization then does not depend on the mesh at all.
6. **Messages need their own storage and their own freeing.** Every message in
   flight is state: a queue entry, a counter, an event value, a callback. On an
   infinite mesh the number of messages in flight is not bounded, so this
   state is not bounded, and it is freed by other messages. Pages are freed by
   release when the dependent's rows are stamped. That state is bounded by the
   pages, and the pages were needed anyway.
7. **A speculative message lets a function run before its data.** "The pages
   will arrive" is not "the pages have arrived". A function that runs on the
   promise reads stale pages and writes a wrong output with a good stamp. The
   digest will disagree later, and the NFE will be repeated, but the work was
   wasted and the wrong output already travelled a hop.
8. **Fan-in multiplies all of the above.** A function whose inputs come from
   many Minis would need a message from each, each a hop, each a race with its
   pages. The rows give one check for all of them: are they all stamped k.

The rule that follows is the one already in the spec. A Mini acts on data,
and only on data, and acts the moment the data is present. The row's stamp is
the only signal. The GPU is handed only functions whose rows are stamped. The
NFE concludes on a status word, never on a message inside the data. With that
rule a Mini's latency and utilization are the same whether it has one
neighbour or an infinite mesh behind it. Without it they are not.
