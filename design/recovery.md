# Recovery: a link's and a call's recovery from loss

**STATUS: CRITICAL PRIORITY.** The operator's decisions of 2026-10-14 settle its shape (mandates 11-14): an execution
fails, returning an error that names what it lost, when its link topology changes or an operand is invalidated, and
nothing resumes it; the caller may run it again at once, on the map as it is then. The node's resident bridge pairs its
links, keeps them paired and pairs them again, with or without a client (design/bridge-and-ipc.md). What remains open
is listed under Requirements and Risks.

Operator, 2026-10-06: "lets split out reconnection and resumption logic into a completely different module which
has never been fully defined in terms of the rubric of all requirements and all mandates. there are many different
kinds of network problems which can happen in differnt link topologies, some which are not testable in the live
sense until certain other real life conditions come to pass; we should treat this as a critical level priority
which is unfinished, unsettled, improperly reviewed, but nevertheless is allowed to operate unchanged in the
interim".

## Scope

Recovery is everything between a loss and the mesh doing its work again, at every layer:

| layer | owns | where |
|---|---|---|
| bridge (link) | loss by events, failing the programs on a lost link, retiring a lost port's device, pairing again on events; the mesh's map; forwarding programs along their routes | `rdma/mesh-flow.c`, `rdma/mesh-verbs.h` |
| client (libmesh) | its bridge's exit watched: its own words cancelled and its starts failed (ESRCH) when the bridge process exits | `rdma/mesh-dataflow.c` `mesh_watch` |
| rank | no detector of its own: its bridge's cancel ends its waits, and a failed window ends the call | metal-microbench `mesh_rank.m` |
| driver (calls) | a call is a request to resident members, on the bridge's map taken at its start; a revoke is a cancel request; shrink and extend run on the bridges' map | metal-microbench `tools/mesh/grid.py`, `docs/elastic.md` |
| collectives library (sessions) | no clock of its own: the bridge's events end its waits (design/bridge-and-ipc.md#programs); a session finished (Finalize), closed in order (Destroy) or as an abort (Abort, Revoke, a failure), and opened again | `rdma/nccl-mesh.c`, `rdma/NCCL.md` |
| port (physical) | a fabric port whose cable link is up with no RDMA peer: bounced at each keeper pass unless its link is the node's only one, logged | `bin/mesh-fabric-init.sh` (every keeper pass) |

## Mandates

Verbatim operator statements outrank everything else (metal-microbench AGENTS.md). Collected 2026-10-06 with their
sources; the ones that bear on what is built are first.

1. **Segmentation is the regression test** (first recorded 2026-10-05; mb docs/elastic.md, mb docs/measurement.md):
   "one of the essential and motivating regression tests of this repository is that the network gets segmented
   extremely often in a manner which would never affect aggregate performance if pairing and renegotiation of mesh
   connections was taken seriously".
2. **Across calls** (2026-10-01; mb docs/principles.md): "the *mesh* applicaiton backend is to recover from any error
   *across* calls, *diachronically*, not have a bunch of overwrought bullshit to pretend *within-call* state is
   recoverable, *synchronically*."
3. **Between NFEs** (2026-09-15; design/deliverables.md): "rdma links on thunderbolt are chuddy. add more requirements
   about recovering after network segmentation of the form seen in cable disconnects / reconnects (two peers exist,
   but where the thunderbolt bridge goes on each side is a new logical/indexed connection between two peers who can
   very quickly reestablish who they are and what they'll be doing. recovery only has to happen between NFEs; there
   is no error recovery guarantee within a NFE.)"
4. **Availability** (THREAT-MODEL.md, 2026-08-27): "the *absolute* requirement for this project is that machines may
   not become unreachable and may never 'choose' to turn off or stop executing rdma chained intermediates or
   results"; "the most important class of compromise is the false negative: the computer should be on and receiving
   work and doing work, but isn't, for any reason".
5. **Nodes come and go** (mb docs/principles.md): nothing in scope makes mesh units "permanent, stationary, stay online
   forever"; nobody stops working when a node is "unplugged, turned off, turned on, retopoed"; "a retopo should not
   involve rewriting any source code from first principles".
6. **No precommitted routing** (2026-10-03; GOAL.md): "i cannot and will not precommit to a specific ordering or
   routing!"; every algorithm holds for every membership and topology, read from an explicit link map. Also ring
   days and star days (mb principles.md): "fast both on days when the 5 computers are a ring and also on days when
   the 5 computers are a star".
7. **No slower transport** (2026-10-01, mb docs/measurement.md): "there will be no excuses for making the transport
   slower; and no algorithms can require a slower transport api".
8. **No TCP double-validation** (2026-10-01, mesh 70338a8): "somebody keeps trying to implement tcp to double-validate
   that the sent memory has been sent (even though the rdma api itself gives you this feedback at the driver level)".
9. **Measurement** (mb docs/measurement.md): "token equality is not important nor is anything like logit equality. we
   measure divergences"; no investigation above 10 s of intensive workload unless asked.
10. **Recovery is required** (2026-10-14, after the M5's RDMA port stayed down for hours under a live cable link): "connection
   recovery on spotty hardware (rdma over thunderbolt is NOT trustworthy at the physical substrate level) is required,
   rather than optional".
11. **Link maps across executions** (2026-10-14, T4): "link maps have to be re-resolvable across function executions which use
   the mesh headers. it's okay for a function call which uses colelctives in some call graph to fail insteado f returning
   if the link topo changed mid function execution, but any loop which uses mesh lib fucntion executions should be able to
   handle different linkmaps across different function executions smoothly".
12. **Operands** (2026-10-14, T5): "functions can fail from any one operand becoming invalidatd, servers should not fail when
   any operand is invalidated (ideally)".
13. **No within-call resumption** (2026-10-14, D2): "you already found substantial evidence that within call failures cannot
   be resume and instead must return an error (which a caller can presumably retry against immediately if that's what
   they want)".
14. **Paired without a client** (2026-10-14, D3), on whether the bridge keeps links paired with no client attached: "this
   emerges from other requirements. required."

**The tension, resolved.** Mandates 2 and 3 place recovery between calls; mandate 1 asks that segmentation never affect
aggregate performance. The resumption inside a call that mesh 828e41c built (a link suspended and paired again on new
queue pairs at what each side had landed) was removed by D2 (mandate 13): a lost link fails the executions on it, and
aggregate performance across a segmentation is the next execution's to keep, its setup kept short by the resident
bridge (no bridges or probes per call).

## Requirements

Each with its source and the mechanism that meets it at the end state (design/bridge-and-ipc.md, rdma/mesh.h,
rdma/mesh-plan.h, rdma/nccl.h); **open** where none is decided. Which of them source holds, and what has run, is
git's and output_data's to say (AGENTS.md: assignments expire), not this table's.

| id | requirement | source | mechanism |
|---|---|---|---|
| Q1 | A drop of a direct link costs the executions on it, which fail as values; the next pairs on the link again | mandates 1, 11, 13 | a loss fails each flow on the link (`#loss`); the port pairs again at once and at the next pairing event (`#pairing`) |
| Q2 | A long drop costs no more than the work the remaining members could not do | mandates 1, 4 | open: the survivors' next executions run on the map without the lost node |
| Q3 | Recovery between executions only; none inside one | mandates 2, 3, 13 | a failed flow is never resumed; the next program is a new flow on the map as it is then (design/bridge-and-ipc.md#flows) |
| Q4 | A link down when a program starts fails the program at once; the link pairs again on its own | mandates 1, 4 | `#first` (ENETDOWN, EHOSTUNREACH); the port's turns (`#pairing`) |
| Q5 | Peer identity independent of the link: the node and the pairing's index | mandate 3 | the hello carries the node; the port table counts pairings (design/bridge-and-ipc.md#ports); a node's records are ordered by their sequence, so a bridge started again moves past its last one (`#records`) |
| Q6 | Loss is observed at the completion queue and the control socket, never by a data-path timer | mandates 3, 8 | by events alone (`#detection`) |
| Q7 | The bridge pairs again while it is up, whether or not a client is attached | mandates 3, 14 | every port's turn at each pairing event, for the bridge's life (`#pairing`); no neighbour warm-up |
| Q8 | A realized program survives link loss without being realized again | mandate 3 | dropped by mandate 13: the execution fails and is run again |
| Q9 | A port move needs no configuration | mandates 3, 5, 11 | the bridge finds each port's peer itself (design/bridge-and-ipc.md#ports) |
| Q10 | A partition of more than two nodes is a topology change, not an error storm | mandates 1, 6 | each side's executions across the cut fail once, naming the node (`#loss`); each side's next runs on its map (`#the-drivers-side`) |
| Q11 | A cable pull and replug, three times, with time to repair and to the first completed execution recorded | mandates 1, 3, 10 | a measurement on hardware, its record in output_data |
| Q12 | Recovery costs nothing on the healthy path | mandate 7 | the data path reads no recovery state |
| Q13 | No per-transfer TCP; TCP at pairing only | mandate 8 | the control socket carries a program's pairing messages and the map's records, nothing per transfer |
| Q14 | No server blocks on an operand; no completion queue is waited on without an event | RDMA-RULES.md, mandate 12 | the bridge's loop sleeps in kevent until an event; a port's progress threads park on their port's bell, looking again every 1 ms while anything can land for them (the provider runs no completion channel) and waiting on the bell alone otherwise (design/bridge-and-ipc.md#progress); a control socket never blocks: a message queues behind what the socket has not taken and goes on at its write event (mesh-flow.c port_say, port_flush); a teardown moves its queue pairs to the error state and destroys them, waiting on nothing the move flushes (mesh-verbs.h `down_pair`); a client's start waits on its status word, which its bridge's pairing or failure, a cancel or a revoke, or the bridge's exit wakes (every wait and what ends it: design/bridge-dataflow.md#waits-and-their-ends). A kernel verbs call (opening a device, reading the device list, querying a port, making or destroying a queue pair, registering memory) can sleep in the driver (RDMA-RULES.md): the bridge makes them only to open, find, pair and tear down a port's device, never per transfer, each on that port's own queue, so one blocks only that port; the event loop, the other ports and clients' starts go on (design/bridge-and-ipc.md#flows) |
| Q15 | Port loss retires all device state; an ordinary reconnection keeps context, PD and registrations | RDMA-KERNEL-RECOVERY.md | a link off or a detached interface retires the device, and no hello or connection reaches the port until its turn has retired the device and opened it again (design/bridge-and-ipc.md#ports) |
| Q16 | No cross-queue GPU spin; every spun-on word in a canceller's set | memory no_cross_queue_gpu_spins | a failed flow cancels its client's words toward the node past their landed counts, after its last landing (prepared-machine.md M12, design/bridge-and-ipc.md#control); where the bridge itself exits, the client cancels its own words (mesh-dataflow.c `mesh_watch`); a GPU wait that reads a cancel halts its program's later waits, so no wait outlives a cancel and none is cancelled twice |
| Q19 | A link that answers its peer's hello but not RDMA does not flap at call level | mandate 1 | open: such a link's programs wait for their pairing, and a paired one's for their landings, until a loss event or the caller's revoke |
| Q20 | Algorithms total over every membership and topology (ring, star, relays, uncabled pairs) | mandate 6 | the map is every algorithm's operand (mesh-plan.h `struct mesh_link_map`), and members its links leave apart are joined along routes the bridges forward (`mesh_map`) |

## The module

### detection

A loss is an event (design/bridge-dataflow.md#4-failure-cancellation-and-release): a link-off or detach of the port's
own interface (M24), the control socket's end of file or reset, a send on it failing, a peer's FAREWELL (its links lost
at once, before its socket's end), an error completion or a refused post on one of the link's queue pairs, a PAIRS
entry that carries a code (the peer's program on that lease lost), a client's process exit (EVFILT_PROC: its program
retired as an abort), and a revoke or cancel (the program's slot retired). An error on a lease's queue pair is that
lease's program's: the program fails as a value, its other leases told the code. An error on the keepalive's queue
pair, or a completion queue's poll failing, is the link's loss. A loss cancels the words past their landed counts of
every program it fails (`#loss`), ending every wait on them. Where the bridge process
itself exits, each client process's own watch of it (mesh-dataflow.c `mesh_watch`, EVFILT_PROC on the bridge's pid)
cancels that process's words, fails its programs' statuses (ESRCH) and wakes its waits, so no GPU program spins on a
word nobody can store.

The map's prune of a node no paired path reaches fails nothing by itself: a program rides leases of its own links and
its relays' links, and fails as those do.

No clock decides a loss. A peer that stays silent with its link, its socket and its bridge up (its client's program
never runs its side, its bridge stopped by a signal) is no event: the waits on it end at the caller's revoke or cancel
(nccl.h `ncclCommRevoke`, `ncclMeshCommCancel`; mesh-dataflow.c `mesh_retire`). The keepalive frames on a link's queue
pair 0 keep its Thunderbolt from going quiet under a program, and detect nothing.

### loss

A link's loss (mesh-flow.c `port_lost`) fails every program holding one of its leases: the link down and every lease
word `MESH_LEASE_DOWN`; each such program failed as a value naming the peer (ECONNRESET where the peer said farewell,
else the loss's code), its words past their landed counts cancelled whether or not its client still holds its slot
(its rows stay the program's until it is freed), and its leases on other links made again in place with the code told
their peers, whose programs fail alike; the link's queue pairs, all of them together, moved to the error state and
destroyed and its two completion queues destroyed, under the port's guard
(`link_unmake`); its request channels ended (`requests_lost`), and the relays and meetings riding them with them. A
loss of the port itself (link off, interface detached) also retires the port's device (registrations, protection domain
and context) on its queue at the port's next turn, which opens it again, as a new bridge would open it, once its link
is on. A verbs call that fails while a queue pair is made at the link's pairing, made again in place or connected is
the link's loss with the port's device retired too (closed, and opened again with the window registered on it before
the port pairs again); a PAIRS whose count is under two (ENOSPC) loses the link with the device kept. A device that
fails otherwise with its link active is retired and opened again at once (`port_opened`).

A lease's own failure (an error completion or a refused post on its queue pair) is its program's: the program fails as
a value (domain 2, naming this node), every lease it holds is made again in place and the code told to the peer
(PAIRS), which fails the peer's program on that lease; the link stays paired and its other leases run on. A lease that
fails with no program on it is made again in place and told with the code. A lease's pair is never destroyed while its
link is up: it is made again in place (mesh-verbs.h `pair_reset`: the error state, which flushes its posted receives and
SENDs, then the reset state and INIT, its number kept and its round one more), and waits in INIT, taking nothing, until
the peer tells the same round. The link's queue pairs are destroyed only all together, at its loss or the stop, each
moved to the error state and then destroyed under the port's guard (mesh-verbs.h `pair_destroy`): nothing waits on what
the move flushes, and a destruction the driver refuses is logged and its handle let go.

### close

A program that ends in order is not a failure: `mesh_detach` (libnccl-mesh's `ncclCommDestroy`, after
`ncclCommFinalize` finished its GPU work) leaves its slot to its bridge, which posts every cell the program released and
takes those SENDs' completions (RDMA's own report that they were sent), then lets go of each lease: its queue pair made
again in place and told to the peer with code 0 and `sent`, the SENDs completed on its making before. The bridge holds the
program's rows until then, and the client waits on none of it. The peer's program holding that lease lets go of it once
its ring has landed `sent` (mesh-flow.c `peer_released`): its status toward that destination stopped, ECONNRESET naming
the far node, its words kept, so the client reads the end before its next group (a held session fails that group as a
value and opens again at the one after: nccl-mesh.c `session_watch`, `session_ensure`). A relay passes an end in order
on: the side that let go is released once its landings are in, the other side's SENDs of them complete, and that side
lets go in its turn with its own `sent`; with neither side held the relay has ended, so no relay outlives its route's
programs. The root of an execution, which holds its relays' request channels, writes a byte past each request at its
own end in order (`mesh_detach`), and the relays' programs then end only as their neighbours let go; a channel that
closes without it ends them as an abort. On the loopback fabric (rdma/mesh-loop.c `destroy`, mesh 2594a087), 20 runs of
every member destroying right after its last group, on the host path and on the walk, on a line of 4 and a torus of 6,
cancelled no landing: every program and every relay ended in order (80 and 80 a path on the line, 120 and 120 on the
torus) and no program failed (metal-microbench output_data/dead-history-20261017/scratch-rewrite/all.txt).

### farewell

A bridge that stops sends FAREWELL on every link before closing it, carrying why it stops (the signal it took, or its
binary changed), as far as its socket takes it; its peers' programs on those links fail with ECONNRESET (the
propagation of an orderly end), each peer logs the link "parted" with that reason (never "lost"), and their links pair
again when the bridge is back. A FAREWELL the socket cannot take leaves the peer the socket's end: the same loss,
logged "lost". Before its FAREWELLs the stopping bridge ends its request channels (ESHUTDOWN): the relays whose
requester has not said it ends in order fail, their codes told on the links still up.

### diagnosis

The operator, 2026-10-15, after the Mac mini restarted at 16:22 with nothing of ours running on it: "ensuring this class
of error is diagnosable from mesh logs (e.g. we can sync critical importance events over the rdma link s.t. a failing
node tells its neighbors along the mesh bridge *how* it is beefing it, if it is beefing it, even if it cannot
self-diagnose or self-correct beefing-it-ness)", and "a crash could spread contagiously across a mesh by a computer
shutting down for unknown reasons *during* a DMA operation, or cause kernel errors which persist across the lifespan
of affected peer processes". Finding that restart's cause took an hour by hand: the bridge logs had no times, nothing
on the M5 held the Mini's last state, and the reason was only in the Mini's power controller's record (a supply loss,
`por,por_vddrtc`, shared with a GPU workstation on its circuit and two monitors).

- Every bridge log line carries the wall clock (UTC, ms) and the node's uptime (mesh-verbs.h `say`).
- Each node's record carries its health (mesh-plan.h `struct mesh_health`): when it was sampled; the node's boot time
  and boot session; why it last restarted, from `kern.shutdownreason` (its power controller's records newest first,
  the 80 bytes the sysctl keeps: `por,...` its supply was lost, `wdog,...` its watchdog reset it; any other record is
  "other", none "unknown", never a guess), and the reset's own records in `shutdown`, whole (the ring's from its newest
  up to the next that opens an earlier boot's, a `por,por_vddrtc` or `wdog` record after the first: mesh-flow.c
  `reset_read`), so a sag reads in them: an undervoltage record (`uv,...`) newest, the always-on rail holding (the Mac
  mini's reset of 2026-10-15 20:44, `uv,por_warn uv,vdd_boost_uvlo por rst`), or after a boot's five `por,por_vddrtc`
  records, the supply sagging before it was lost (the operator's reading: the records between a boot's five and the
  previous boot's are how the supply went); a sag where the reset's own records hold `uv` (mesh-plan.h
  `MESH_RESET_SAG`); its power source (AC, battery or UPS, with charge and minutes to empty: a USB UPS is a power source
  macOS reports); free and wired memory and the kernel's memory pressure; thermal pressure. The bridge samples it at
  each turn of its event loop from cheap reads (sysctl, VM statistics, the thermal notify state); the power sources'
  description is read at each of their notifications. A change of power source, of charge off AC, or of memory or
  thermal pressure (each a notification the loop takes: the power sources' and thermal pressure's notify names, the
  memory-pressure dispatch source) is logged and sent at once; the rest rides the record, which is flooded again at the
  configuration's `health_us` interval (etc/bridge/README.md).
- Records flood (bridge-and-ipc.md#records), so every bridge holds every node's last health. A change of health alone
  is written under the region's health serial (`hdr.health`), never the map serial: the map serial moves only where the
  mesh changes, and a communicator's session runs on the map its members agreed on until it fails, its next session
  reading the map then (nccl-mesh.c `session_ensure`, `map_digest`); sampling is no change of the mesh.
- A survivor logs, at the moment of loss, the node it lost (or that parted, with its reason), and, when the map drops a
  node no paired path reaches, that node's last health. A node that comes back is logged with its new health and
  whether it is a new boot (its boot session changed); a peer's urgent change is logged as it arrives ("node N (name)
  health: ...").
- `mesh-stat` prints every node's health (`nodes`), so the telemetry service and `mesh-top` carry it; node-info prints
  `last_reset` (the newest power-controller record).
- A supply loss gives the node itself no warning: its last words are the health its neighbours hold, and the reason is
  read at its next boot. A UPS on its USB turns that into a warning the mesh hears (power UPS, its charge falling).

### records

A node's records are ordered by their sequence, the greater bytes newer among equal sequences (RFC 2328 s13.1's
checksum), with no clock in them (design/bridge-and-ipc.md#records). A bridge that starts again begins its sequence
anew: where a bridge hears its own node's record newer than the one it holds (an earlier start's, flooded back as a
port pairs), it moves its sequence past it and floods its own again (s13.4); where a bridge hears a record older than
the newest it has of that node, map or not, it sends the newer back on the port it came from (s13 step 8), so a node
whose earlier records every neighbour still remembers learns them at its first pairing. A record no newer than the
newest heard is dropped and not sent on, so a record the map pruned never circulates.

### registration

A port's device registers the whole window when it comes up (design/prepared-machine.md#M09), checked against that
device's max_mr on the port's own queue, before the link makes any queue pair. A registration the device refuses (the
window's extents past its max_mr, EFBIG, or an extent `ibv_reg_mr` refuses) is logged with the extent, its bytes, the
errno and the extents the device holds; the port keeps those extents and turns again at its next memory-pressure change
(the dispatch source the event loop takes) or link event (its link on, its interface attached), registering the rest. A
node of several cables registers the window once a device (80 extents each at an 80 GiB window), never yet run on more
than one device of a node, so a refusal of the second device is the first thing to look for when a Studio's second
cable will not pair.

### pairing

A port turns (its device opened where it is not and the window registered on it, then its HELLO said to every node on
its link) at events: its finding, its link coming on or its interface attaching, a paired link lost (the port says
HELLO again at its next turn: a link that had paired pairs again as fast as its peer answers), the peer's HELLO (the
lower node dials the higher's listener on the address it heard, the higher answers a HELLO so the lower hears it at
once), a memory-pressure change after a refused registration, and the configuration's `hello_us` interval for a port
with no peer (a HELLO lost on its link has no other event; a paced emission, etc/bridge/README.md). On the control
socket the two ends' HELLOs pair the link: each end makes the link's two completion queues and its queue pairs (as many
as its device allows, at most 17: 10 on TB5, the keepalive's and nine leases) and tells the peer their numbers (PAIRS,
TN3205's one out-of-band step); at the peer's PAIRS each pair is joined to the peer's of its index, the leases are free,
and the link is up. Those queue pairs live for the link's life; a lease's is made again in place (reset, its number
kept) only where its holder lets go of it or fails, the peer told its new round. A port whose device failed with its link active is retired and opened
again at once, and where that fails too, at the next of those events. A connection that never completed ends at the
kernel's own connect error.

### first

A program's members first agree its leases and relays (mesh-call.h `mesh_transfers_agree`): each reports the leases it
claimed on its paired links to the `meet` service of the first member's bridge, which holds every report until all the
members have sent one; the first member routes every pair of members on its own bridge's map, takes for each hop a
lease of each queue that both of the hop's ends hold free, asks each relay on a route to forward the pair (its bridge's
`relay` service), and tells every member its leases. The agreement fails at once as a value on every member where the
root's plan refuses: EHOSTUNREACH (no route on its map, or a member or relay with no paired cable to the next hop),
EBUSY (no lease free on a hop, or a relay's lease taken), a relay's refusal, ECONNRESET (a member's meeting cut short),
EINVAL. A member that never reports holds every member's agreement until its caller's revoke or cancel
(`mesh_transfers_interrupt`), or until the meeting's channel ends (a link on its route lost, a bridge on it stopping).

A started program waits on no peer: its bridge compiles its chains on the leases its slot names and posts its first
receives within its event loop's turn, or fails it at once as a value (mesh-call.h `mesh_transfers_start`): ENETDOWN
(the lease's link not paired), ENETRESET (a lease no longer the program's: the link paired again since the agreement),
EINVAL (a list or lease it cannot carry), ENOMEM; ECANCELED where its own slot is retired while it waits (a revoke),
ESRCH where its bridge exits (mesh-dataflow.c `mesh_watch`). A SEND released before the far program's receives are
posted waits in the hardware (TN3205's credit), so members that start apart need no handshake. A peer whose program
never starts is waited for, by the landings this program's waits read, until the caller's revoke or cancel.

An execution's routes are its root's: the members' and relays' leases come from one plan made on one map, whatever map
each member read. Members whose programs are made from different maps meet at their session's first position:
libnccl-mesh publishes each rank's digest of its programs' making there and fails the session at once as a value where
they differ (nccl-mesh.c `session_agree`: "the members' maps differ"), so a caller runs the execution again on the map
as it is then.

### the driver's side

Each execution reads its node's bridge's map at its start (mesh-plan.h `mesh_map`; no probes, no bridges of its own),
and its routes are its root's bridge's map at the agreement; a link the bridges pair again, or a route through other
nodes, is in the next execution's map.

### stop

A bridge whose binary changes (EVFILT_VNODE on the binary it runs: its contents written, or the file replaced, renamed
or removed) stops at once, as a signal stops it: SIGTERM, SIGINT and SIGHUP set one stop flag, which every loop of the
bridge reads every pass (design/bridge-dataflow.md#waits-and-their-ends). The stop joins the ports' progress threads,
ends the request channels (ESHUTDOWN), tells each port's peer FAREWELL and loses the link, so every program it carries
fails as a value on both ends (its words past their landed counts cancelled) and every queue pair is moved to the error
state and destroyed; it then tells the region the bridge is gone (`bridge_pid` 0), retires each port's device and
exits. launchd starts the new binary, which posts its start (mesh-plan.h `MESH_STARTED`, `mesh_bridge_wait`); a client
sees the old one's exit through its watch of it (`mesh_watch`). Each caller's next call runs on the new bridge. On the
loopback fabric (rdma/mesh-loop.c `kill ... binary`, mesh 2594a087), a node's binary replaced by rename under a call
stopped its bridge on a line of 4 and on a torus of 6: every rank's call failed as a value, the bridge started again
and the map converged each time, and the NCCL matrix ran exact after it. `mesh-loop sigterm` stops every bridge in each
of eight states (idle, pairing, host, walk, relay, linkloss, client, peer): 32 of 32 readings exited 0 with no region
and no queue pair left, each stop within 6 ms (metal-microbench output_data/dead-history-20261017/scratch-rewrite/all.txt).

### port

A Thunderbolt port can come back from a peer's reboot or a link flap with its cable link up and its RDMA port down
(PORT_DOWN on the M5 for hours on 2026-09-23 and 2026-10-14; a reseat fixed the first). Every keeper pass, the fabric
init reads each port's link and whether the IORegistry holds an `AppleThunderboltRDMAPeerInterface` for it, never
opening a verbs context (RDMA-KERNEL-RECOVERY.md). A port with its link up and no peer is bounced at each pass (its
interface down, then up) until the peer appears, unless its link is the node's only one; each bounce, and the recovery
with its duration and count, goes to `fabric.log` and the unified log, and `mesh-health` warns while it lasts.

## Risks and defects found in review so far

1. **Immediate re-pairing and the September panic.** mesh be9731b (2026-09-28): "A lost session waits 3 s before
   pairing again: failed completions, queue-pair teardown and immediate re-pairing preceded the Sep 5 kernel panic."
   A paired link lost pairs again at once; a link's loss and the stop move all its queue pairs to the error state and
   then destroy them under the port's guard; the cause of the panic itself is not established.
2. **Neighbour cache after a replug.** RDMA-RULES.md: after any replug both sides fail RTR unless the neighbour cache
   is warmed. Not done; the hellos on the link may warm it; a pull may show otherwise.
3. **Arrival counts reset by their last arriver** (mb docs/kernels.md#decode-attention, 2026-10-08): the last-arrival
   merges leave a count nonzero when a dispatch is cut, and a later run of the same program merges wrongly without
   failing. No path here runs a program again after a cut: a failed execution's process prepares its programs again,
   their counts starting at zero.
4. **The error state before destruction on Apple's provider is unread.** Every teardown moves its queue pairs to the
   error state and then destroys them (mesh-verbs.h `pair_destroy`), waiting on nothing the move flushes (Apple's
   provider flushes fewer completions than a pair held: 687 of 688 on the M5 at mesh 2e02569a), so no provider holds a
   port's queue at a teardown; whether a destruction right after the move leaves the device clean on hardware is
   unread, and a destruction the driver refuses is logged and its handle let go. The guard's pause of a port's progress
   threads while a lease's pair is made again in place is unmeasured. Integration reads both first
   (RDMA-KERNEL-RECOVERY.md).
5. **A peer silent with its link up.** A peer whose client never runs its side, or whose bridge is stopped by a signal
   with its link and socket up, is no event: the waits on it end at the caller's revoke (`#detection`).
6. **A pair made again in place while the link's others run.** The M5 bridge's crash of 2026-10-17
   (RDMA-KERNEL-RECOVERY.md "October 16 and 17") was a post on a long-lived queue pair 1.5 s after another queue pair of
   its device was destroyed. No queue pair of a link is destroyed while the link is up: a lease's pair is made again in
   place (mesh-verbs.h `pair_reset`: the error state, the reset state, INIT, its number kept) while the keepalive pair
   and the other leases run, and the link's pairs are destroyed only together, at its loss or the stop. Whether Apple's
   provider takes a reset and a reconnection of one pair while the others post is unread on hardware.
7. **A context that refuses every connection.** After the Mac mini's kernel panic of 2026-10-17 its device refused
   every INIT to RTR until a new process opened it (RDMA-KERNEL-RECOVERY.md). A verbs call refused while a pair is made,
   made again in place or connected loses the link with the port's device retired: closed, and opened again by the same
   process before the port pairs again. Whether a context opened again within the bridge's process connects where the
   refusing one did not is unread on hardware.
8. **An orderly end toward a program that never received.** A program ending in order waits for its SENDs'
   completions, and TB5 holds a SEND until the peer's receive is posted. A peer that agreed and never attached its
   receives (its start refused before attaching, its process gone between the agreement and its start) tells nothing on
   the lease, so the ending program holds its leases and its slot until the link's loss or the stop
   (design/bridge-dataflow.md#waits-and-their-ends); the root's plans meanwhile skip those leases.

## Failures by kind and topology

Topologies: **P** the TB5 pair; **R** a ring or complete mesh of 3-6 nodes; **U** uncabled pairs routed through other nodes' bridges; **H** heterogeneous links (TB5, TB4, Ethernet); **Q** programs
of several queues (a lease each, up to four a destination).

| # | failure | what happens |
|---|---|---|
| F1 | a direct link down mid-call | the programs holding its leases fail, naming the peer, and their codes fail their other members and relays; the link pairs again as it comes back (`#pairing`) |
| F3 | a link down when a program starts | its agreement finds no paired cable on the hop (EHOSTUNREACH); a start after the agreement fails at once (ENETDOWN, or ENETRESET where the link paired again since) |
| F6 | the peer bridge stops or crashes | its farewell or the socket's end: the programs on the link fail; the link pairs again with the new bridge; a bridge stopped by a signal with its socket up is no event (Risk 5) |
| F7 | the peer's program fails or ends | a failure's code on the lease fails this side's program (its words past their landed counts cancelled); an end in order stops this side's status toward it (ECONNRESET naming it) once its last SENDs have landed, its words kept (`#close`) |
| F8 | the peer node reboots or loses power | the link-off: the programs on it fail, the port's device retired and opened again |
| F9 | the control plane (LAN, ssh, mDNS) lost, the cable up | the bridges' links are unaffected; the driver's reach is |
| F10 | the data path dead, the control socket alive | an error completion where the provider reports one fails the program on that lease; else no event: the program's waits end at the caller's revoke (Risk 5) |
| F11 | one direction lost | undefined |
| F12 | a link degrades (speed, errors) | the Balancer on evidence |
| F13 | one link of several drops | the programs holding its leases fail; the others' links are untouched |
| F14 | a relay bridge on a route drops | its links' loss at its neighbours fails the programs holding those leases, naming the relay node, and their codes the route's ends; the next execution's route is the current map's |
| F15 | a cable moved to another port or peer | the old port's programs fail; the new port pairs with whatever node it reaches |
| F18 | the driver's own node fails | the stream stops (one driver) |
| F19 | a node of several cables stops in a mesh whose survivors hold a cycle | its last record ends at the first bridge that has heard it (`#records`) |
| F20 | a lease's queue pair fails (an error completion, a refused post) | its program's alone (`#loss`): the program fails, the lease is made again in place and its code told, the link and its other leases run on; the keepalive pair's failure, or a completion queue's, loses the link; a verbs call refused while a pair is made again or connected loses the link and retires its device (Risk 7) |
| F21 | a device refuses the window's registration | logged with the extent; the port turns again at its next memory-pressure change or link event (`#registration`) |
| F22 | two ends of a routed pair whose maps differ | one route, the root's (`#first`); a relay on it with no paired cable to a neighbour refuses (EHOSTUNREACH), and every member's agreement fails as a value |
| F23 | members' programs of one execution made from maps a change apart | the session's first position carries each rank's digest: the session fails at once on every member ("the members' maps differ", `#first`) |
| F24 | a relay refuses its forwarding (a lease taken, no cable, no memory) | the root's assignment carries its code: every member's agreement fails as a value and frees its claims |
| F25 | a member of an execution never reports to its meeting | every member's agreement waits until its caller's revoke or cancel, or the meeting's channel ends (`#first`) |
| F26 | the node's own bridge exits under a client's waits | the client's watch of it cancels the client's words and fails its starts (ESRCH): its GPU programs end and its calls fail as values |
| F27 | a member gone between the agreement and its start | its peers' programs wait for its landings until their callers' revoke; a peer ending in order toward it holds its leases until the link's loss or the stop (Risk 8) |

## The review this asks for

1. The device-state rules and the immediate re-pairing (Risks 1, 2, 4, 6, 7) checked with a pull and replug on
   hardware, once runtime is allowed.
2. Q2 and Q19 decided, or dropped; Q11 and F1-F15 run where they can be.
3. Risk 8 decided: the wait it leaves ends at nothing a caller does.
