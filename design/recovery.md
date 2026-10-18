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
| collectives library (sessions) | no clock of its own: the bridge's events end its waits (design/bridge-and-ipc.md#control); a session finished (Finalize), closed in order (Destroy) or as an abort (Abort, Revoke, a failure), and opened again | `rdma/nccl-mesh.c`, `rdma/NCCL.md` |
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
| Q14 | No server blocks on an operand; no completion queue is waited on without an event | RDMA-RULES.md, mandate 12 | the bridge's loop sleeps in kevent until an event; a port's progress threads park on their port's bell, looking again every 1 ms while anything can land for them (the provider runs no completion channel) and waiting on the bell alone otherwise (design/bridge-and-ipc.md#progress); a control socket never blocks: a message queues behind what the socket has not taken and goes on at its write event (mesh-flow.c port_say, port_flush); a teardown's flush drains exactly the work requests it held (mesh-verbs.h `pair_flush`); a client's start waits on its status word, which its bridge's pairing or failure, a revoke, or the bridge's exit wakes. A kernel verbs call (opening a device, reading the device list, querying a port, making or destroying a queue pair, registering memory) can sleep in the driver (RDMA-RULES.md): the bridge makes them only to open, find, pair and tear down a port's device, never per transfer, each on that port's own queue, so one blocks only that port; the event loop, the other ports and clients' starts go on (design/bridge-and-ipc.md#flows) |
| Q15 | Port loss retires all device state; an ordinary reconnection keeps context, PD and registrations | RDMA-KERNEL-RECOVERY.md | a link off or a detached interface retires the device, and no hello or connection reaches the port until its turn has retired the device and opened it again (design/bridge-and-ipc.md#ports) |
| Q16 | No cross-queue GPU spin; every spun-on word in a canceller's set | memory no_cross_queue_gpu_spins | a failed flow cancels its client's words toward the node past their landed counts, after its last landing (prepared-machine.md M12, design/bridge-and-ipc.md#control); where the bridge itself exits, the client cancels its own words (mesh-dataflow.c `mesh_watch`); a GPU wait that reads a cancel halts its program's later waits, so no wait outlives a cancel and none is cancelled twice |
| Q19 | A link that answers its peer's hello but not RDMA does not flap at call level | mandate 1 | open: such a link's programs wait for their pairing, and a paired one's for their landings, until a loss event or the caller's revoke |
| Q20 | Algorithms total over every membership and topology (ring, star, relays, uncabled pairs) | mandate 6 | the map is every algorithm's operand (mesh-plan.h `struct mesh_link_map`), and members its links leave apart are joined along routes the bridges forward (`mesh_map`) |

## The module

### detection

A loss is an event (design/bridge-and-ipc.md#control): a link-off or detach of the port's own interface (M24), the
control socket's end of file or reset, a send on it failing, a peer's FAREWELL (its flows lost at once, before its
socket's end), a FAIL a flow's peer sends on its control channel (or on its lanes, a lane rider's), an error
completion, a failed post or poll, a client's process exit (EVFILT_PROC: its program retired as an abort), a revoke,
and the map's prune of a node no paired path reaches (a program awaiting that node's READYs fails, EHOSTUNREACH). A
failed completion, post or poll is its flow's alone, and fails its client program's every flow. A loss cancels the
program's words past their landed counts (`#loss`), ending every wait on them. Where the bridge process itself exits,
each client process's own watch of it (mesh-dataflow.c `mesh_watch`, EVFILT_PROC on the bridge's pid) cancels that
process's words, fails its programs' statuses (ESRCH) and wakes its waits, so no GPU program spins on a word nobody
can store.

No clock decides a loss. A peer that stays silent with its link, its socket and its bridge up (its client's program
never runs its side, its bridge stopped by a signal) is no event: the waits on it end at the caller's revoke
(nccl.h `ncclCommRevoke`, mesh-dataflow.c `mesh_retire`).

### loss

A link's loss fails every flow on it: its client's words on the link past their landed counts cancelled (its program's
every flow muted first, design/bridge-and-ipc.md#control), its status the loss's code (ECONNRESET where the peer said
farewell), its queue pairs destroyed, and then the link's lanes (design/bridge-and-ipc.md#lanes), once neither
progress thread reads them. A lane's failure (a completion's error, a header out of step) is its lanes' alone, and so
are lanes that could not be made (a verbs call's error; their pages are leased for every port at the bridge's start
and their two queue pairs counted before any flow's own, and a device or window that has no room for them yet is
waited for, never refused): the flows that ride them and have posted on them fail, the lanes are closed and made again
at once, the peer told (a LANES of none and why; its own lanes, joined to these, are reset alike), and the link stays
paired, its flows with queue pairs of their own and those still pairing running on (`lanes_reset`). Lanes that fail to
connect lose the link and retire the port's device, which opens again as a new bridge would open it (a context that
refuses every queue pair's INIT to RTR, as the Mac mini's did after its 2026-10-17 kernel panic, connects once opened
again: metal-microbench output_data/split-20261017). A loss of the port itself (link off, interface detached) also
retires the port's device: registrations, protection domain and context, once no flow and no lane holds a queue pair
on it. A flow's own failure is told to the peer (FAIL: an exclusive flow's on its control queue pair by its send
thread, the flow stopping once that SEND's completion is taken or the peer's own FAIL came; a lane rider's on its
lanes), which fails its side. A flow's queue pairs are moved to the error state and drained of exactly the work
requests they held before they are destroyed, under the port's guard, whether it failed or closed.

### close

A program that ends in order is not a failure: `mesh_detach` (libnccl-mesh's `ncclCommDestroy`, after
`ncclCommFinalize` finished its GPU work) leaves its bridge to post every cell it released and take those SENDs'
completions (RDMA's own report that they were sent), then release its flows, holding the program's rows until then;
nothing is told to the peer, and the client waits on none of it. A peer's landings are its own: it releases its side
when its own client detaches, having had every landing its program waited on. A lane rider that ends in order tells
the next hop RELEASED; a relay passes it on toward the route's other end and releases its two flows once both ends'
came (each end had every cell it waited on and sent every one it released), so no relay outlives its route's programs.
A peer that ended in order and makes its program of the execution again reaches this side as a second READY on a flow
that heard one already: that flow fails as the peer's end (ECONNRESET naming it, its words past their landed counts
cancelled; a held session fails its group in flight as a value and opens again at its next group) and the READY is
taken as a new one (mesh-flow.c `ready_heard`), so recovery between calls reaches a held communicator whose peer
started its program again; a peer that never comes back leaves a held session's next group waiting until the
caller's revoke.
On the loopback fabric's line of 4 (rdma/mesh-loop.c `destroy`), 180 runs of every member destroying right after its
last group, host path and walk at 4,096, 65,536 and 1,048,576 elements, lost no landing; every endpoint flow ended in
order, every relay was released by both its ends, and every flush drained exactly its count
(metal-microbench output_data/dead-history-20261017/scratch-delete/w1).

A port's stripe flow (design/bridge-dataflow.md#stripes) fails as its lanes do: alone (`stripes_reset`), the requests it
held failing as values (each writer's word `UINT64_MAX`, its ring's code), the peer told (a STRIPES of none and why),
and made again as a request needs it; one that could not be made or connected fails the next request toward its peer
as a value by that code, and the one after makes it again; a link lost closes it with the lanes. Its teardown moves
its queue pairs to the error state and drains them before destroying them, and every verbs change on a port holding
one runs under the port's guard (design/RDMA-KERNEL-RECOVERY.md "The stripe writes of October 1").

### farewell

A bridge that stops sends FAREWELL on every link before closing it, carrying why it stops (the signal it took, or its
binary changed), as far as its socket takes it; its peers' programs on those links fail with ECONNRESET (the
propagation of an orderly end), each peer logs the link "parted" with that reason (never "lost"), and their links pair
again when the bridge is back. A FAREWELL the socket cannot take leaves the peer the socket's end: the same loss,
logged "lost".

### diagnosis

The operator, 2026-10-15, after the Mac mini restarted at 16:22 with nothing of ours running on it: "ensuring this class
of error is diagnosable from mesh logs (e.g. we can sync critical importance events over the rdma link s.t. a failing
node tells its neighbors along the mesh bridge *how* it is beefing it, if it is beefing it, even if it cannot
self-diagnose or self-correct beefing-it-ness)", and "a crash could spread contagiously across a mesh by a computer
shutting down for unknown reasons *during* a DMA operation, or cause kernel errors which persist across the lifespan
of affected peer processes". Finding that restart's cause took an hour by hand: the bridge logs had no times, nothing
on the M5 held the Mini's last state, and the reason was only in the Mini's power controller's record (a supply loss,
`por,por_vddrtc`, shared with a GPU workstation on its circuit and two monitors).

- Every bridge log line carries the wall clock (UTC, ms) and the node's uptime (mesh-flow.c `say`).
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
- Records flood as before (bridge-and-ipc.md), so every bridge holds every node's last health. A change of health
  alone is written under the region's health serial (`hdr.health`), never the map serial: the map serial moves only
  where the mesh changes, and a communicator's session reads the map again where it moved and reopens only where the
  change changes its own routes or their links' costs (nccl-mesh.c `map_digest`, `session_ensure`); sampling is no
  change of the mesh.
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
device's max_mr on the port's own queue. A registration the device refuses (the window's extents past its max_mr,
EFBIG, or an extent `ibv_reg_mr` refuses) is logged with the extent, its bytes, the errno and the extents the device
holds; the port keeps those extents and turns again at its next memory-pressure change (the dispatch source the event
loop takes) or link event (its link on, its interface attached), registering the rest. A node of several cables
registers the window once a device (80 extents each at an 80 GiB window), never yet run on more than one device of a
node, so a refusal of the second device is the first thing to look for when a Studio's second cable will not pair.

### pairing

A port turns (its device opened where it is not, then its HELLO said to every node on its link) at events: its finding,
its link coming on or its interface attaching, a paired link lost (at once: a link that had paired pairs again as fast
as its peer answers), the peer's HELLO (the lower node dials the higher's listener on the address it heard, the higher
answers a HELLO so the lower hears it at once), a memory-pressure change after a refused registration, and the
configuration's `hello_us` interval for a port with no peer (a HELLO lost on its link has no other event; a paced
emission, etc/bridge/README.md). A port whose device failed with its link active is retired and opened again at once,
and where that fails too, at the next of those events. A connection that never completed ends at the kernel's own
connect error.

### first

A program started on a link that is not paired fails at once (ENETDOWN). A program waits for its peer's program to
start, and for its queue pairs to pair, on events alone: its peer's READY, PAIRS and POSTED, a FAIL, a link lost, its
own slot retired (a revoke: ECANCELED), its bridge's exit (ESRCH, mesh-dataflow.c `mesh_watch`). A peer whose program
never starts is waited for until the caller's revoke. A program toward a node its map gives no route fails at once
(EHOSTUNREACH) at either end: the lower end makes the route, and the higher end, which awaits the lower's READYs, is
refused at once where its own map holds no route to the lower, and fails as a value (EHOSTUNREACH, naming the lower)
where the map then prunes the lower.

A routed program's start returns once the far member's program has started and every relay between has posted (POSTED
is end to end, design/bridge-and-ipc.md#protocol), so a member that starts later than its peer delays the peer's start,
as over a direct cable. Its route is its lower node's (design/bridge-and-ipc.md#routes): the higher end awaits the
lower's READYs whatever its own map's route (it reads no route of its own to pair), so two ends whose maps differ at
their starts still pair along one route; a routed program's start returns once every one of its stripes runs
(design/bridge-and-ipc.md#cables). Two ends that disagree on whether a cable joins them (one sees it paired, the other
not yet) fail at once as values (EPROTO, naming the node that refused). Members whose programs of an execution are made
from different maps (their digests differ, mesh-call.h `mesh_transfers_digest`) fail at once, every destination of each
as a value (ESTALE, naming the node whose program differs), wherever a READY of one reaches the other, so a caller may
bind only the members its programs move with and run the execution again on the map as it is then
(design/bridge-and-ipc.md#protocol).

### the driver's side

Each execution reads its node's bridge's map at its start (mesh-plan.h `mesh_map`; no probes, no bridges of its own);
a link the bridges pair again, or a route through other nodes, is in the next execution's map.

### stop between calls

A bridge whose binary changes (EVFILT_VNODE on the binary it runs: its contents written, or the file replaced, renamed
or removed) sets its region's `stopping`, which a client holding flows past their execution reads (mesh-plan.h
`mesh_stopping_wait`), and stops once it carries no flow and no awaiting program; launchd starts the new binary, which
posts its start (mesh-plan.h `MESH_STARTED`, `mesh_bridge_wait`). No call is ended for it.

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
   A paired link lost pairs again at once; every teardown is flushed under the port's guard; the cause of the panic
   itself is not established.
2. **Neighbour cache after a replug.** RDMA-RULES.md: after any replug both sides fail RTR unless the neighbour cache
   is warmed. Not done; the hellos on the link may warm it; a pull may show otherwise.
3. **Arrival counts reset by their last arriver** (mb docs/kernels.md#decode-attention, 2026-10-08): the last-arrival
   merges leave a count nonzero when a dispatch is cut, and a later run of the same program merges wrongly without
   failing. No path here runs a program again after a cut: a failed execution's process prepares its programs again,
   their counts starting at zero.
4. **The flush on Apple's provider is unread.** A move to the error state and its flushed completions have never run on
   hardware; every teardown drains exactly the work requests it held (mesh-verbs.h `pair_flush`), so a provider that
   flushes fewer holds that port's queue at the teardown; the guard's pause of a port's progress threads during another
   flow's verbs work is unmeasured. Integration reads both first (design/bridge-dataflow.md#stripes,
   RDMA-KERNEL-RECOVERY.md).
5. **A peer silent with its link up.** A peer whose client never runs its side, or whose bridge is stopped by a signal
   with its link and socket up, is no event: the waits on it end at the caller's revoke (`#detection`).

## Failures by kind and topology

Topologies: **P** the TB5 pair; **R** a ring or complete mesh of 3-6 nodes; **U** uncabled pairs routed through other nodes' bridges; **H** heterogeneous links (TB5, TB4, Ethernet); **Q** links of
several queue pairs (the torch/NCCL sessions: four).

| # | failure | what happens |
|---|---|---|
| F1 | a direct link down mid-call | the executions on it fail; the link pairs again as it comes back (`#pairing`) |
| F3 | a link down when a program starts | the program fails at once (ENETDOWN) |
| F6 | the peer bridge stops or crashes | its farewell or the socket's end: the executions fail; the link pairs again with the new bridge; a bridge stopped by a signal with its socket up is no event (Risk 5) |
| F7 | the peer's program fails or ends | FAIL fails this side's program (its words past their landed counts cancelled); an end in order tells nothing (`#close`): a program of this side still waiting on the ended one waits until the peer's program of the execution says READY again (ECONNRESET, then taken anew) or its caller's revoke |
| F8 | the peer node reboots or loses power | the link-off: the executions fail, the port's device retired |
| F9 | the control plane (LAN, ssh, mDNS) lost, the cable up | the bridges' links are unaffected; the driver's reach is |
| F10 | the data path dead, the control socket alive | no event: the program's waits end at the caller's revoke (Risk 5) |
| F11 | one direction lost | undefined |
| F12 | a link degrades (speed, errors) | the Balancer on evidence |
| F13 | one link of several drops | the executions on it fail; the others' links are untouched |
| F14 | a relay bridge on a route drops | the programs routed through it fail, naming the relay node; the next execution's route is the current map's |
| F15 | a cable moved to another port or peer | the old port's executions fail; the new port pairs with whatever node it reaches |
| F18 | the driver's own node fails | the stream stops (one driver) |
| F19 | a node of several cables stops in a mesh whose survivors hold a cycle | its last record ends at the first bridge that has heard it (`#records`) |
| F20 | a link's lanes fail (a lane's completion error, a header out of step, a make refused) | the lanes' alone (`#loss`): the flows that posted on them fail, both ends make new lanes at once, the link stays paired and its own queue pairs run on; lanes that fail to connect lose the link, its device retired and opened again |
| F21 | a device refuses the window's registration | logged with the extent; the port turns again at its next memory-pressure change or link event (`#registration`) |
| F22 | two ends of a routed pair start on maps a flap apart | both take the lower end's route (design/bridge-and-ipc.md#routes); a cable one end sees and the other does not fails both at once (EPROTO) |
| F23 | members' programs of one execution made from maps a change apart | the READYs carry each program's digest: every member's program fails at start (ESTALE, `#first`) |
| F24 | a routed start whose relays' own jobs end before the next hop's POSTED | each relay tells POSTED as its partner hears it (design/bridge-and-ipc.md#protocol) |
| F25 | a stripe flow fails (a completion's error, a descriptor out of the window, the peer's reset) | its own alone: the requests it held fail as values, it is closed (flushed) and made again as a request needs it; the link, its lanes and flows run on |
| F26 | the node's own bridge exits under a client's waits | the client's watch of it cancels the client's words and fails its starts (ESRCH): its GPU programs end and its calls fail as values |

## The review this asks for

1. The device-state rules and the immediate re-pairing (Risks 1, 2, 4) checked with a pull and replug on hardware, once
   runtime is allowed.
2. Q2 and Q19 decided, or dropped; Q11 and F1-F15 run where they can be.
