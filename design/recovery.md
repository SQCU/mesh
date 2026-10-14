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
| bridge (link) | detection, failing the programs on a lost link, retiring a lost port's device, pairing again a pause apart | `rdma/mesh-flow.c`, `rdma/mesh-verbs.h` |
| rank | the silent-link detector, cancelling a link's waits, a failed window ending the call | metal-microbench `mesh_rank.m` `await_window` |
| driver (calls) | revoke, agree, shrink, extend, on the map the bridges record | metal-microbench `tools/mesh/grid.py`, `docs/elastic.md` |
| collectives library (sessions) | `MESH_REMOTE_BOUND`, a session retired and opened again | `rdma/nccl-mesh.c`, `rdma/NCCL.md` |
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

Each with its source and its status as of 2026-10-14, read from source (no runtime: the operator's prohibition of
2026-10-14): **built** (in source, not yet run), **met** (on evidence named), **partial**, **unmet**, **unreachable**
(no live test can reach it until a condition exists).

| id | requirement | source | status |
|---|---|---|---|
| Q1 | A drop of a direct link costs the executions on it, which fail as values; the next pairs on the link again | mandates 1, 11, 13 | built: a loss fails each flow on the link (`#loss`), the port pairs again a pause later (`#pause`) |
| Q2 | A long drop costs no more than the work the remaining members could not do | mandates 1, 4 | partial: survivors continue alone on the map the bridges record |
| Q3 | Recovery between executions only; none inside one | mandates 2, 3, 13 | built: no suspension or resumption exists |
| Q4 | A link down when a program starts fails the program at once; the link pairs again on its own | mandates 1, 4 | built: a flow on an unpaired port fails ENETDOWN (`#first`) |
| Q5 | Peer identity independent of the link: the node, its bridge's boot, and the pairing's index | deliverables R1 | built: the hello carries node and boot; the link table counts pairings |
| Q6 | Loss is observed at the completion queue and the control socket, never by a data-path timer | deliverables R2 | partial: the control socket's keepalive is a timer on the control path, not the data path |
| Q7 | A bounded loop pairs again while the bridge is up, whether or not a client is attached | deliverables R3, mandate 14 | built: every port's turn, a pause apart, for the bridge's life (`#pause`); no neighbour warm-up |
| Q8 | A realized program survives link loss without being realized again | deliverables R4 | dropped by mandate 13: the execution fails and is run again |
| Q9 | A port move needs no configuration | deliverables R5, mandate 11 | built: the bridge finds each port's peer itself (design/bridge-and-ipc.md#ports) |
| Q10 | A partition of more than two nodes is a topology change, not an error storm | deliverables R6 | unreachable: two nodes on hardware |
| Q11 | A cable pull and replug, three times, with time to repair and to the first completed execution recorded | deliverables R7 | unmet: not run |
| Q12 | Recovery costs nothing on the healthy path | mandate 7 | built: the data path reads no recovery state |
| Q13 | No per-transfer TCP; TCP at pairing only | mandate 8 | built: the control socket carries a program's pairing messages and nothing per transfer |
| Q14 | Every blocking call bounded; no unbounded completion-queue poll | RDMA-RULES.md, mandate 12 | built: the bridge's loop waits 250 ms at most; pairing has its windows; sends wait 1 s; a teardown that fails keeps its handles for the next turn; a client's start waits on the bridge, which bounds it, and on the bridge's life |
| Q15 | Port loss retires all device state; an ordinary reconnection keeps context, PD and registrations | RDMA-KERNEL-RECOVERY.md | built: a link off or a detached interface retires the device before the port opens it again |
| Q16 | No cross-queue GPU spin; every spun-on word in a canceller's set | memory no_cross_queue_gpu_spins | met: a failed flow cancels its client's words on the link |
| Q19 | A link that answers its peer's hello but not RDMA does not flap at call level | mandate 1 | partial: such a link fails its executions' pairing (ETIMEDOUT within the pairing window) |
| Q20 | Algorithms total over every membership and topology (ring, star, relays, uncabled pairs) | mandate 6 | unreachable on hardware (two nodes) |

## The module as built

### detection

A loss is a link-off or detach of the port's own interface (M24), the control socket ending (a farewell before its end
makes it an orderly one), its keepalive failing, or a send on it failing. A failed completion, post or poll is its
flow's alone. The control socket's keepalive (1 s idle, 1 s interval, 2 probes) tells a side whose own interface stays
up (its peer's went down, its peer's host died): UC queue pairs report nothing.

### loss

A link's loss fails every flow on it: its threads joined, its client's pending words on the link cancelled, its status
the loss's code (ECONNRESET where the peer said farewell), its queue pairs destroyed. A loss of the port itself (link
off, interface detached) also retires the port's device: registrations, protection domain and context, once no flow
holds a queue pair on it. A flow's own failure is told to the peer (FAIL), which fails its side.

### farewell

A bridge that stops sends FAREWELL on every link before closing it; its peers' programs on those links fail with
ECONNRESET (the propagation of an orderly end) and their links pair again when the bridge is back.

### pause

3 s between a port's turns: between pairings and between hellos, the pause a lost session took after the 2026-09-05
panic (mesh be9731b: "failed completions, queue-pair teardown and immediate re-pairing preceded the Sep 5 kernel panic").
No setting changes it.

### first

A program started on a link that is not paired fails at once (ENETDOWN). A program waits for its peer's program to
start at most 600 s (members prepare their programs at their own speeds: metal-microbench docs/measurement.md
2026-10-10, the verify step), and for its queue pairs to pair at most 30 s from then.

### the driver's side

The driver derives each call's membership and map from what the bridges record (no probes, no bridges of its own); a
link the bridges pair again is in the next call's map.

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
4. **Nothing here has run.** The resident bridge (2026-10-14) is source only, under the operator's prohibition of
   runtime until the acceptance checklist is attained (metal-microbench docs/acceptance-20261014.md).

## Failures by kind and topology

Topologies: **P** the TB5 pair; **R** a ring or complete mesh of 3-6 nodes (Studios: four M5 Ultra arriving, cabling
open); **U** uncabled pairs routed through relay rings; **H** heterogeneous links (TB5, TB4, Ethernet); **Q** links of
several queue pairs (the torch/NCCL sessions: four).

| # | failure | as built |
|---|---|---|
| F1 | a direct link down mid-call | the executions on it fail; the link pairs again a pause after it is back |
| F3 | a link down when a program starts | the program fails at once (ENETDOWN) |
| F6 | the peer bridge stops or crashes | its farewell or the keepalive: the executions fail; the link pairs again with the new bridge |
| F7 | the peer's program fails or ends | END or FAIL: this side's program on the link fails |
| F8 | the peer node reboots or loses power | the keepalive or the link-off: the executions fail, the port's device retired on a link-off |
| F9 | the control plane (LAN, ssh, mDNS) lost, the cable up | the bridges' links are unaffected; the driver's reach is |
| F10 | the data path dead, the control socket alive | the rank's silent bound, call fails |
| F11 | one direction lost | undefined |
| F12 | a link degrades (speed, errors) | the Balancer on evidence |
| F13 | one link of several drops | the executions on it fail; the others' links are untouched |
| F14 | a relay on an uncabled pair's route drops | the executions on its links fail; the next execution's routes are its map's |
| F15 | a cable moved to another port or peer | the old port's executions fail; the new port pairs with whatever node it reaches |
| F18 | the driver's own node fails | the stream stops (one driver) |

## The review this asks for

1. The device-state and pause rules (Risks 1-2) checked with a pull and replug on hardware, once runtime is allowed.
2. Q2, Q6, Q10, Q11, Q19 and Q20 decided and built or dropped; F1-F15 run where they can be.
