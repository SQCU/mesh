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
| bridge (link) | detection, failing the programs on a lost link, retiring a lost port's device, pairing again a pause apart; the mesh's map; forwarding programs along their routes | `rdma/mesh-flow.c`, `rdma/mesh-verbs.h` |
| rank | no silent-link detector of its own: the library's bound ends its waits, and a failed window ends the call | metal-microbench `mesh_rank.m` |
| driver (calls) | a call is a request to resident members, on the bridge's map taken at its start; a revoke is a cancel request; shrink and extend run on the bridges' map | metal-microbench `tools/mesh/grid.py`, `docs/elastic.md` |
| collectives library (sessions) | mesh.h `MESH_SILENT_NS`, the library's one bound; a session retired and opened again | `rdma/nccl-mesh.c`, `rdma/NCCL.md` |
| port (physical) | a fabric port whose cable link is up with no RDMA peer: bounced with backoff, logged | `bin/mesh-fabric-init.sh` (every keeper pass) |

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
| Q1 | A drop of a direct link costs the executions on it, which fail as values; the next pairs on the link again | mandates 1, 11, 13 | a loss fails each flow on the link (`#loss`); the port pairs again a pause later (`#pause`) |
| Q2 | A long drop costs no more than the work the remaining members could not do | mandates 1, 4 | open: the survivors' next executions run on the map without the lost node |
| Q3 | Recovery between executions only; none inside one | mandates 2, 3, 13 | a failed flow is never resumed; the next program is a new flow on the map as it is then (design/bridge-and-ipc.md#flows) |
| Q4 | A link down when a program starts fails the program at once; the link pairs again on its own | mandates 1, 4 | `#first` (ENETDOWN, EHOSTUNREACH); the port's turns (`#pause`) |
| Q5 | Peer identity independent of the link: the node, its bridge's boot, and the pairing's index | mandate 3 | the hello carries node and boot; the port table counts pairings (design/bridge-and-ipc.md#ports) |
| Q6 | Loss is observed at the completion queue and the control socket, never by a data-path timer | mandates 3, 8 | open: the control socket's keepalive is a timer on the control path, and the library's silent bound (`#detection`) one on landed positions |
| Q7 | A bounded loop pairs again while the bridge is up, whether or not a client is attached | mandates 3, 14 | every port's turn, a pause apart, for the bridge's life (`#pause`); no neighbour warm-up |
| Q8 | A realized program survives link loss without being realized again | mandate 3 | dropped by mandate 13: the execution fails and is run again |
| Q9 | A port move needs no configuration | mandates 3, 5, 11 | the bridge finds each port's peer itself (design/bridge-and-ipc.md#ports) |
| Q10 | A partition of more than two nodes is a topology change, not an error storm | mandates 1, 6 | each side's executions across the cut fail once, naming the node (`#loss`); each side's next runs on its map (`#the-drivers-side`) |
| Q11 | A cable pull and replug, three times, with time to repair and to the first completed execution recorded | mandates 1, 3, 10 | a measurement on hardware, its record in output_data |
| Q12 | Recovery costs nothing on the healthy path | mandate 7 | the data path reads no recovery state |
| Q13 | No per-transfer TCP; TCP at pairing only | mandate 8 | the control socket carries a program's pairing messages and the map's records, nothing per transfer |
| Q14 | Every blocking call bounded; no unbounded completion-queue poll | RDMA-RULES.md, mandate 12 | the bridge's loop waits 250 ms at most; a port's progress threads park on their port's bell at most 1 ms at a time while anything is theirs to look for (3 s otherwise, design/bridge-and-ipc.md#progress); pairing has its windows; a control socket never blocks: a message queues behind what the socket has not taken and goes on at its write event (mesh-flow.c port_say, port_flush), and the keepalive (1 s idle, 1 s interval, 2 probes) ends a peer that takes none; a teardown that fails keeps its handles for the next turn; a client's start waits at most the peer and pairing windows and a pause (`MESH_PEER_NS` + `MESH_PAIR_NS` + `MESH_PAUSE_NS`, ETIMEDOUT) and ends at once where the bridge is gone (ESRCH). The one exception: a kernel verbs call (opening a device, reading the device list, querying a port, making or destroying a queue pair, registering memory) cannot be bounded from user space, since it can sleep uninterruptibly in the driver (RDMA-RULES.md): the bridge makes them only to open, find, pair and tear down a port's device, never per transfer, each on that port's own thread, so one blocks only that port; the event loop, the other ports and clients' starts go on (design/bridge-and-ipc.md#flows) |
| Q15 | Port loss retires all device state; an ordinary reconnection keeps context, PD and registrations | RDMA-KERNEL-RECOVERY.md | a link off or a detached interface retires the device, and no hello or connection reaches the port until its turn has retired the device and opened it again (design/bridge-and-ipc.md#ports) |
| Q16 | No cross-queue GPU spin; every spun-on word in a canceller's set | memory no_cross_queue_gpu_spins | a failed flow cancels its client's words toward the node (prepared-machine.md M12) |
| Q19 | A link that answers its peer's hello but not RDMA does not flap at call level | mandate 1 | open: such a link fails its executions' pairing (ETIMEDOUT within the pairing window), or the library's silent bound fails a paired one's (`#detection`); the bridge exchanges nothing of its own over RDMA to tell it |
| Q20 | Algorithms total over every membership and topology (ring, star, relays, uncabled pairs) | mandate 6 | the map is every algorithm's operand (mesh-plan.h `struct mesh_link_map`), and members its links leave apart are joined along routes the bridges forward (`mesh_map`) |

## The module

### detection

A loss is a link-off or detach of the port's own interface (M24), the control socket ending (a farewell before its end
makes it an orderly one), its keepalive failing, or a send on it failing. A failed completion, post or poll is its
flow's alone. The control socket's keepalive (1 s idle, 1 s interval, 2 probes) tells a side whose own interface stays
up (its peer's went down, its peer's host died): UC queue pairs report nothing. A destination with positions pending
that lands none for `MESH_SILENT_NS` (mesh.h, 10 s) is cancelled (`mesh_cancel`), ending every wait on it: the
library's one bound, its session's progress thread the watcher (rdma/nccl.h).

### loss

A link's loss fails every flow on it: its client's pending words on the link cancelled, its status the loss's code
(ECONNRESET where the peer said farewell), its queue pairs destroyed, and then the link's lanes (design/bridge-and-ipc.md
#lanes), once neither progress thread reads them. A lane's failure (a completion's error, a header out of step) is
its lanes' alone, and so are lanes that could not be made or connected (a verbs call's error; their pages are leased
for every port at the bridge's start and their two queue pairs counted before any flow's own, and a device or window
that has no room for them yet is waited for, never refused): the flows that ride them and have posted on them
fail, the lanes are closed and made again a pause (3 s) later, the peer told (a LANES of none and why; its own lanes,
joined to these, are reset alike), and the link stays paired, its flows with queue pairs of their own and those still
pairing running on (`lanes_reset`). No link is lost for its lanes, and none is held unpaired: at the round's first
pass failed lanes stayed failed until the link itself went, and its second pass's first commit (152cf86) lost the link
and held its port for a pause doubling to 30 min, a demotion of an edge the mesh's contract forbids (AGENTS.md). A
loss of the port itself (link off, interface detached) also retires the port's
device: registrations, protection domain and context, once no flow and no lane holds a queue pair on it. A flow's own
failure is told to the peer (FAIL), which fails its side.

### farewell

A bridge that stops sends FAREWELL on every link before closing it, carrying why it stops (the signal it took, or its
binary changed), and waits up to 200 ms for it to leave; its peers' programs on those links fail with ECONNRESET (the
propagation of an orderly end), each peer logs the link "parted" with that reason (never "lost"), and their links pair
again when the bridge is back.

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
  and boot session; why it last restarted, from `kern.shutdownreason` (its power controller's records newest first:
  `por,...` its supply was lost, `wdog,...` its watchdog reset it; any other record is "other", none "unknown", never a
  guess); its power source (AC, battery or UPS, with charge and minutes to empty: a USB UPS is a power source macOS
  reports); free and wired memory and the kernel's memory pressure; thermal pressure. The bridge samples it each turn
  from cheap reads (sysctl, VM statistics, notify states; the power sources' description only when their notify state
  moved). A change of power source, of charge off AC, or of memory or thermal pressure is logged and sent at once; the
  rest rides the record at most 10 s later.
- Records flood as before (bridge-and-ipc.md), so every bridge holds every node's last health. A change of health
  alone is written under the region's health serial (`hdr.health`), never the map serial: the map serial moves only
  where the mesh changes, and a communicator's session reads the map again where it moved and reopens only where the
  change changes its own routes or their links' costs (nccl-mesh.c `map_digest`, `session_ensure`); sampling is no
  change of the mesh.
- A survivor logs, at the moment of loss, the node it lost (or that parted, with its reason), and, when the map drops a
  node no paired path reaches, that node's last health and how long before it was heard. A node that comes back is
  logged with its new health, whether it is a new boot (its boot session changed), and how long it was gone; a peer's
  urgent change is logged as it arrives ("node N (name) health: ...").
- `mesh-stat` prints every node's health (`nodes`), so the telemetry service and `mesh-top` carry it; node-info prints
  `last_reset` (the newest power-controller record).
- A supply loss gives the node itself no warning: its last words are the health its neighbours hold, and the reason is
  read at its next boot. A UPS on its USB turns that into a warning the mesh hears (power UPS, its charge falling).

### registration

A port's device registers the whole window when it comes up (design/prepared-machine.md#M09), checked against that
device's max_mr on the port's own queue. A registration the device refuses (the window's extents past its max_mr,
EFBIG, or an extent `ibv_reg_mr` refuses) is logged with the extent, its bytes, the errno and the extents the device
holds; the port keeps those extents and turns again 60 s later, doubling to 30 min (the fabric keeper's schedule,
`#port`), registering the rest. At MESH_VERSION 112 a refusal was silent, retired the device and registered the whole
window again every 3 s, and the bridge's start opened every device on its main thread to read the least max_mr (a
wedged device stalled every port); a node of several cables registers the window once a device (80 extents each at an
80 GiB window), never yet run on more than one device of a node, so a refusal of the second device is the first thing
to look for when a Studio's second cable will not pair.

### pause

3 s between a port's turns: between pairings and between hellos, the pause a lost session took after the 2026-09-05
panic (mesh be9731b: "failed completions, queue-pair teardown and immediate re-pairing preceded the Sep 5 kernel panic").
No setting changes it.

### first

A program started on a link that is not paired fails at once (ENETDOWN). A program waits for its peer's program to
start at most 600 s (members prepare their programs at their own speeds: metal-microbench docs/measurement.md
2026-10-10, the verify step), and for its queue pairs to pair at most 30 s from then. The client's own wait for its
start is bounded by the same windows and a pause (mesh.h `MESH_PEER_NS`, `MESH_PAIR_NS`, `MESH_PAUSE_NS`, their one
definition), so a bridge that never answers ends it as ETIMEDOUT. A program toward a node its map gives no route fails
at once (EHOSTUNREACH) where it is the pair's lower end; the higher end awaits the lower's route (below), its peer
told of the lower's failure by none, so its wait ends at the peer window or at its own program's revoke.

A routed program's start returns once the far member's program has started and every relay between has posted (POSTED
is end to end, design/bridge-and-ipc.md#protocol), so a member that starts 25 s after its peer delays the peer's start,
as over a direct cable, rather than failing its first group at the library's silent bound. Its route is its lower
node's (design/bridge-and-ipc.md#routes): the higher end awaits the lower's READY within the peer window and fails
ETIMEDOUT naming the lower past it, whatever its own map holds (it reads no route of its own to pair, so a higher end
that sees none at its start still pairs on the lower's); where its map has no route to the lower, it waits the pairing
window (30 s) from when the lower was last reachable (a link of the route pairing again), the peer window once a route
appears, and is refused at once (EHOSTUNREACH, naming the lower) where the lower has been unreachable longer or was
never heard of, so a call toward a dead node fails within 30 s at either end; two ends whose maps differ at their
starts still pair along one route; a routed program's start returns once every one of its stripes runs (design/bridge-and-ipc.md#cables). Two ends
that disagree on whether a cable joins them (one sees it paired, the other not yet) fail at once as values (EPROTO,
naming the node that refused) rather than wait out the peer window. Members whose programs of an execution are made
from different maps (their digests differ, mesh-call.h `mesh_transfers_digest`) fail at once, every destination of
each as a value (ESTALE, naming the node whose program differs), wherever a READY of one reaches the other: none
waits the peer window for a READY no program of the other's takes, so a caller may bind only the members its programs
move with and run the execution again on the map as it is then (design/bridge-and-ipc.md#protocol).

### the driver's side

Each execution reads its node's bridge's map at its start (mesh-plan.h `mesh_map`; no probes, no bridges of its own);
a link the bridges pair again, or a route through other nodes, is in the next execution's map.

### port

A Thunderbolt port can come back from a peer's reboot or a link flap with its cable link up and its RDMA port down
(PORT_DOWN on the M5 for hours on 2026-09-23 and 2026-10-14; a reseat fixed the first). Every keeper pass, the fabric
init reads each port's link and whether the IORegistry holds an `AppleThunderboltRDMAPeerInterface` for it, never
opening a verbs context (RDMA-KERNEL-RECOVERY.md). A port with its link up and no peer for 60 s is bounced (interface
down, 2 s, up), again after 2, 4, 8 and 16 min and then every 30 min, until the peer appears; each bounce, and the
recovery with its duration and count, goes to `fabric.log` and the unified log, and `mesh-health` warns while it lasts.

## Risks and defects found in review so far

1. **Immediate re-pairing and the September panic.** mesh be9731b (2026-09-28): "A lost session waits 3 s before
   pairing again: failed completions, queue-pair teardown and immediate re-pairing preceded the Sep 5 kernel panic."
   Answered by `#pause` (3 s); the cause of the panic itself is not established.
2. **Neighbour cache after a replug.** RDMA-RULES.md: after any replug both sides fail RTR unless the neighbour cache
   is warmed. Not done; the hellos on the link may warm it; a pull may show otherwise.
3. **Arrival counts reset by their last arriver** (mb docs/kernels.md#decode-attention, 2026-10-08): the last-arrival
   merges leave a count nonzero when a dispatch is cut, and a later run of the same program merges wrongly without
   failing. No path here runs a program again after a cut: a failed execution's process prepares its programs again,
   their counts starting at zero.

## Failures by kind and topology

Topologies: **P** the TB5 pair; **R** a ring or complete mesh of 3-6 nodes; **U** uncabled pairs routed through other nodes' bridges; **H** heterogeneous links (TB5, TB4, Ethernet); **Q** links of
several queue pairs (the torch/NCCL sessions: four).

| # | failure | what happens |
|---|---|---|
| F1 | a direct link down mid-call | the executions on it fail; the link pairs again a pause after it is back |
| F3 | a link down when a program starts | the program fails at once (ENETDOWN) |
| F6 | the peer bridge stops or crashes | its farewell or the keepalive: the executions fail; the link pairs again with the new bridge |
| F7 | the peer's program fails or ends | END or FAIL: this side's program on the link fails |
| F8 | the peer node reboots or loses power | the keepalive or the link-off: the executions fail, the port's device retired on a link-off |
| F9 | the control plane (LAN, ssh, mDNS) lost, the cable up | the bridges' links are unaffected; the driver's reach is |
| F10 | the data path dead, the control socket alive | the library's silent bound (`MESH_SILENT_NS`) cancels the destination; the execution fails |
| F11 | one direction lost | undefined |
| F12 | a link degrades (speed, errors) | the Balancer on evidence |
| F13 | one link of several drops | the executions on it fail; the others' links are untouched |
| F14 | a relay bridge on a route drops | the programs routed through it fail, naming the relay node; the next execution's route is the current map's |
| F15 | a cable moved to another port or peer | the old port's executions fail; the new port pairs with whatever node it reaches |
| F18 | the driver's own node fails | the stream stops (one driver) |
| F19 | a node of several cables stops in a mesh whose survivors hold a cycle | its last record ends at the first bridge that has heard it (design/bridge-and-ipc.md#records); at MESH_VERSION 112 it circulated for good, reopening every session |
| F20 | a link's lanes fail (a lane's completion error, a header out of step, a make or connect refused) | the lanes' alone (`#loss`): the flows that posted on them fail, both ends make new lanes a pause later, the link stays paired and its own queue pairs run on |
| F21 | a device refuses the window's registration | logged with the extent; the port turns again 60 s later, doubling to 30 min (`#registration`) |
| F22 | two ends of a routed pair start on maps a flap apart | both take the lower end's route (design/bridge-and-ipc.md#routes); a cable one end sees and the other does not fails both at once (EPROTO) |
| F23 | members' programs of one execution made from maps a change apart | the READYs carry each program's digest: every member's program fails at start (ESTALE, `#first`), through MESH_VERSION 114 one bound to a member whose program did not bind it back waited 630 s |
| F24 | a routed start whose relays' own jobs end before the next hop's POSTED | each relay tells POSTED as its partner hears it (design/bridge-and-ipc.md#protocol); through MESH_VERSION 114 such a relay never told it and both ends waited 630 s, in a third of the routed starts of a transcription of the code |

## The review this asks for

1. The device-state and pause rules (Risks 1-2) checked with a pull and replug on hardware, once runtime is allowed.
2. Q2, Q6 and Q19 decided, or dropped; Q11 and F1-F15 run where they can be.
