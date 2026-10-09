# Recovery: a link's and a call's recovery from loss

**STATUS: CRITICAL PRIORITY. Unfinished, unsettled, not properly reviewed.** The module this defines
(`rdma/mesh-recovery.h`, with the bridge's side in `rdma/mesh-flow.c` and the driver's in metal-microbench
`tools/mesh/grid.py`) runs as built in the interim (mesh 828e41c, 33a9d35, 3c6fc6a, 2cc452a; mb grid.py `Probes`).
Running is not acceptance: nothing here is settled until the review this document asks for (below) is done, and
the mandates below disagree with part of what is built.

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
| bridge (link) | detection, suspension, pairing again, resumption at what landed, giving up | `rdma/mesh-recovery.h`, `rdma/mesh-flow.c` (M24), `rdma/mesh-verbs.h` |
| rank | the silent-link detector, cancelling a link's waits, a failed window ending the call | metal-microbench `mesh_rank.m` `await_window` |
| driver (calls) | revoke, agree, shrink, extend; probing failed links; preparing ahead | metal-microbench `tools/mesh/grid.py`, `docs/elastic.md` |
| collectives library (sessions) | `MESH_REMOTE_BOUND`, a session retired and opened again | `rdma/nccl-mesh.c`, `rdma/NCCL.md` |

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

**The tension, unresolved.** Mandates 2 and 3 place recovery between calls or between NFEs, with no guarantee inside
one; mandate 1 asks that segmentation never affect aggregate performance. What is built resumes a link *inside* an
NFE, at the records each side landed, so a call survives a short outage: it serves 1 and goes past 3, and it is the
kind of within-call machinery 2 rejected (the lost-session design of mesh 6a9ec69 and 476f15f, removed by the
2026-10-01 revert 3981ca8, resumed inside calls too). Across-call recovery alone meets 2 and 3 and, today, misses 1:
a membership change costs about 23-33 s of call setup (mb docs/elastic.md). Which way the module goes is the
operator's decision; until then it runs as built.

## Requirements

Each with its source and its status as of 2026-10-06: **met** (on evidence named), **partial**, **unmet**,
**conflict** (a mandate disagrees), **unreachable** (no live test can reach it until a condition exists).

| id | requirement | source | status |
|---|---|---|---|
| Q1 | A short drop of a direct link costs the outage, not the call | mandate 1 | met for a pair, interface down 3-8 s (`#suspension`); exact: the collectives matrix, 1548/1548 calls equal to their expected values through a 5 s and a 3 s drop, one resumption re-sending 256 records |
| Q2 | A long drop costs no more than the work the remaining members could not do | mandate 1, 4 | partial: survivors continue alone; readmission costs two call setups (23-33 s each) |
| Q3 | Recovery between NFEs only; none promised inside one | mandates 2, 3 | conflict: resumption works inside an NFE |
| Q4 | A link down at a call's first pairing is retried within the pairing window | mandate 1, 4 | built (`#first`, a pause apart); a drop as the bridges start rode through (the dials waited), the port-down path itself not yet hit live |
| Q5 | Peer identity independent of the link: (node, boot nonce, program epoch) plus a connection index | deliverables R1 | unmet: no instance identity; a restarted peer bridge answers EPROTO |
| Q6 | Loss is observed at the completion queue and the control socket, never by a data-path timer | deliverables R2 | partial: the control socket's keepalive is a timer on the control path, not the data path |
| Q7 | A bounded loop pairs again while the bridge is up: devices enumerated again, the neighbour cache warmed, one deadline an attempt, a fixed pause, the listener reopened | deliverables R3 | partial: bounded by the window; listener reopened; a fixed 3 s pause (`#pause`); no re-enumeration, no neighbour warm-up |
| Q8 | A realized program survives link loss without being realized again | deliverables R4 | met for a pair within the window |
| Q9 | A port move needs no configuration: the configuration names peers, not devices | deliverables R5 | unmet: a link is its device; a moved cable gives up to the call |
| Q10 | A partition of more than two nodes is a topology change, not an error storm | deliverables R6 | unreachable: two nodes on hardware; loopback has no fault injection |
| Q11 | A cable pull and replug, three times, with time to repair and to the first completed NFE recorded | deliverables R7 | unmet: only `ifconfig` down/up so far |
| Q12 | Recovery costs nothing on the healthy path | mandate 7 | to measure: the send thread now reads the provider's queue pair by index (one indirection a post) |
| Q13 | No per-transfer TCP; TCP at pairing (and resumption) only | mandate 8 | met: the landed counts cross once a resumption |
| Q14 | Every blocking call bounded; no unbounded completion-queue poll | RDMA-RULES.md | met by reading (window, pairing deadline, `link_halt` drains what is queued) |
| Q15 | Port loss retires all device state; an ordinary reconnection keeps context, PD and registrations | RDMA-KERNEL-RECOVERY.md | unmet: a link-off keeps context, PD and registrations (see Risks) |
| Q16 | No cross-queue GPU spin; every spun-on word in a canceller's set | memory no_cross_queue_gpu_spins | met: a link's words stay in the rank's cancel set through a suspension |
| Q17 | A rank does not cancel a link its bridge is resuming within the window | consistency of `#window` | met: the detector passes a MESH_SUSPENDED link (mb mesh_rank.m); an 8 s drop: one call, resumed after 6.1 and 8.7 s |
| Q18 | The driver forwards the recovery window to every bridge | consistency | met: grid.py forwards MESH_RESUME_SECONDS and MESH_RECOVERY_PAUSE_MS |
| Q19 | A readmitted link that answers ICMP but not RDMA does not flap at call level | mandate 1 | partial: a link failing within 30 s of its readmission is probed at twice its delay (to 60 s); the probe pings interfaces whose port is active; untested |
| Q20 | Algorithms total over every membership and topology (ring, star, relays, uncabled pairs) | mandate 6 | unreachable on hardware (two nodes); untested on loopback |

## The module as built

### window

`MESH_RESUME_SECONDS` (default 10, the ranks' silent bound; 0 gives up at once, the behaviour before 828e41c).
The window is how long an in-shader wait is held: a rank's GPU waits on a link's words spin until they land or are
cancelled, so a longer window means longer GPU spins (Risks).

### detection

A loss is a link-off of the link's own device (M24), the control socket ending without the peer's farewell, a
failed completion, a failed post or poll. The control socket's keepalive (1 s idle, 1 s interval, 2 probes) tells a
side whose own interface stays up (its peer's went down, its peer's host died): UC queue pairs report nothing.

### loss

The first loss is kept (its code and domain); a later one while the link recovers is the same loss. Given up, it is
taken once and the link stops (M12 cancellation, as before).

### farewell

An orderly end (`link_close`) sends one byte, 0x46, before shutting the control socket; a control socket that ends
with it stops the peer's link, one that ends without it is a loss.

### resumption

On new queue pairs, every record and request stays where `link_configure` laid it. A ring's records land in the
order posted, so each side tells the other its rings' landed counts (`MESH_RESUME` and one count a queue pair),
posts its rings again from its own, and moves each SEND stream back to the peer's count of its queue pair, mid-cell
where a chain landed in part. A barrier then holds every SEND until the peer's receives are posted. A peer that
answers a fresh pairing (a restarted bridge) is EPROTO: no resumption.

### suspension

`mesh_recovery_resume`: the phase `MESH_SUSPENDED`; the transport halted (threads joined, the receives that landed
taken) and released (control socket, listener, queue pairs); then `verbs_up` again until it pairs, the window
passes, or the call abandons the link (the bridge stopped, its client exited, another client took the region, a
rank cancelled the link), a pause before every attempt; then watched, restarted, `MESH_PAIRED`.

### pause

`MESH_RECOVERY_PAUSE_MS` (default 3000) before every pairing again: the pause a lost session took after the
2026-09-05 panic (mesh be9731b: "failed completions, queue-pair teardown and immediate re-pairing preceded the Sep 5
kernel panic"). A listener waits for its peer's dial on poll, not a spinning core.

### first

A first pairing whose port is not active (ENETDOWN, before any queue pair exists) tries again a pause apart within
its pairing window, so a link down as a call starts does not fail it.

### the driver's side

A failed link is probed every 2 s (`Probes`: the pair's interfaces pinged) and readmitted once it answers, which
revokes the call running without it. A failed call also sends the Balancer's parts back (`rates.reject`).

## Risks and defects found in review so far

1. **Immediate re-pairing and the September panic.** mesh be9731b (2026-09-28): "A lost session waits 3 s before
   pairing again: failed completions, queue-pair teardown and immediate re-pairing preceded the Sep 5 kernel panic."
   Answered by `#pause` (3 s); the cause of the panic itself is not established.
2. **Device state on port loss.** RDMA-KERNEL-RECOVERY.md: port loss retires all device state (QP, CQ, MR, PD,
   context, in that order); the resumption keeps context, PD and registrations across a link-off. Worked for an
   interface down/up; untested for a pull, a detach (`KEV_DL_IF_DETACHED`) or a moved port.
3. **Neighbour cache after a replug.** RDMA-RULES.md: after any replug both sides fail RTR unless the neighbour cache
   is warmed. Not done; `ifconfig` did not need it; a pull may.
4. **The data-validity condition.** A transfer's payload slot for invocation t is reused at t + depth (M01's ring,
   depth 2 on the pair). Ordinary operation needs the slot intact until the NIC reads it; a resumption re-sends from
   it, so it needs the slot intact until the peer *landed* it. That holds where the producer of t + depth waits,
   directly or through its own receives, on a word the peer publishes only after landing t: every lock-step program
   (a decode step's crossings both ways; a session whose calls need the peer's data of the call before). It is not
   shown for a producer that can run ahead of its consumer by the depth (a pipeline's prefill chunks, a one-way
   stream): there a lost, unlanded record's slot may hold a later invocation, and the resumption would re-send it.
   Evidence for lock-step: the matrix's second resumption moved a stream back 256 records and every call matched.
5. **The token evidence was degenerate.** The first flap runs' 4096 tokens equal the undisturbed run's, but the
   prompt's answer ends at step 24 and steps 28 onward repeat one token. The exact check is now the collectives
   matrix (Venues); a divergence on non-degenerate generation is still owed.
6. Answered 2026-10-06: the rank's detector racing the window (Q17), the window not forwarded (Q18), the accept
   loop spinning (poll). Partly: readmission hysteresis (Q19).
7. **Stale statements elsewhere**: mb docs/shelf.md and elastic.md call the beacon "the one source of membership
   events", which `Probes` contradicts; design/algorithm-sources.md ("not detected by inventing a timer") against the
   keepalive and the silent bound.
8. **Arrival counts reset by their last arriver** (mb docs/kernels.md#decode-attention, 2026-10-08): the last-arrival
   merges (the engine's decode attention and argmax, the recorder's part publications, LiteRT's native kernels) leave a count nonzero
   when a dispatch is cut, and a later run of the same program merges wrongly without failing. No path here runs a
   program again after a cut: suspension and resumption stop no GPU command, a cancelled wait runs its kernel to the end,
   a failed window ends the call and its rank process, and shrink, extend and readmission are new processes whose
   counts start at zero. Within-call recovery that ran a failed window again (Q3) would have to make them again.

## Failures by kind and topology

Topologies: **P** the TB5 pair (live); **R** a ring or complete mesh of 3-6 nodes (Studios: four M5 Ultra arriving,
cabling open); **U** uncabled pairs routed through relay rings (queue pairs 2-3); **H** heterogeneous links
(TB5, TB4, Ethernet); **Q** links of several queue pairs (the torch/NCCL sessions, MESH_QPS 2 or 4).

| # | failure | as built | P | R/U/H/Q |
|---|---|---|---|---|
| F1 | a direct link down briefly (< window) mid-call | both ends suspend and resume | done: `ifconfig` 3-8 s; exact under the matrix | Q: done (2 queue pairs) |
| F2 | a direct link down past the window | gives up; survivors continue; probe readmits | done: 20 s | untested |
| F3 | a link down at first pairing | fails at once (Q4) | seen 2026-10-05 | untested |
| F4 | a second drop while pairing again | the same window continues | untested | untested |
| F5 | repeated drops | a fresh window each, no backoff | two drops | untested |
| F6 | the peer bridge crashes | gives up (no identity: Q5) | untested since the revert | untested |
| F7 | the peer rank crashes | the peer's farewell: stops | untested live | untested |
| F8 | the peer node reboots or loses power | keepalive, window, give up; beacons | the 2026-10-05 outage, before the module | untested |
| F9 | the control plane (LAN, ssh, mDNS) lost, the cable up | undefined for the driver | untested | untested |
| F10 | the data path dead, the control socket alive | the rank's silent bound, call fails | needs injection | needs injection |
| F11 | one direction lost | undefined | needs injection | needs injection |
| F12 | a link degrades (speed, errors) | the Balancer on evidence | untested | untested |
| F13 | one link of several drops | each link recovers alone; collectives across it wait | unreachable | loopback with injection; Studios |
| F14 | a relay on an uncabled pair's route drops | per-link resumption; routes undefined | unreachable | Studios |
| F15 | a cable moved to another port or peer | gives up (Q9); the driver finds interfaces afresh | untested | Studios |
| F16 | a cyclic session (M30) | landed counts run on across cycles | done: the matrix's sessions, exact | — |
| F17 | an outage longer than a GPU wait may be held | the window bounds it | untested past 10 s in-call | — |
| F18 | the driver's own node fails | the stream stops (one driver) | by design | — |

## Venues

- **Live pair**: the Mini's passwordless sudo takes its Thunderbolt interface down and up (`ifconfig en3`, ssh on the
  LAN); metal-microbench `output_data/segment-20261006/flaps.sh OUT step:seconds ...`.
- **Exact checks under drops**: the collectives matrix (metal-microbench `tools/nccl_demo.py pair`: every call
  checked against its expected values, cyclic sessions, two queue pairs) with drops injected:
  `output_data/recovery-20261006/matrix-flaps.sh TAG t:d ...`; undisturbed and with a 5 s and a 3 s drop, 1548/1548
  calls pass. Still owed: a non-degenerate generation's divergence against the undisturbed run.
- **Loopback** (to build): `mesh-flow-loop` over `loopverbs.c` runs N bridges on one host over any link-map family;
  a per-link fault in the fabric (drop, one direction, stall, sever) would reach F4-F14 and Q10, Q20 without hardware.
- **Hardware not yet here**: a physical pull and replug (Q11), port moves (F15), three or more nodes on TB5 (Studios),
  heterogeneous links.

## The review this asks for

1. The operator's decision on the tension (Q3): within-NFE resumption kept, bounded, or removed in favour of
   between-NFE recovery made cheap.
2. A proof, or a counterexample, of the data-validity argument (Risk 4) for every program kind.
3. The device-state and pause rules (Risks 1-3) reconciled with what is built, with a pull and replug on hardware.
4. Evidence that measures what it claims (Risk 5).
5. Q4, Q5, Q9, Q17-Q19 decided and built or dropped; the loopback injection built; F4-F16 run where they can be.
