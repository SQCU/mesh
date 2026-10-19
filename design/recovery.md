# Recovery: a link's and a call's recovery from loss

**STATUS: CRITICAL PRIORITY.** The operator's decisions of 2026-10-14 settle its shape (mandates 11-14): an execution
fails, returning an error that names what it lost, when its link topology changes or an operand is invalidated, and
nothing resumes it; the caller may run it again at once, on the map as it is then. The operator's rule of 2026-10-18
bounds every use of a DMA or network channel at five seconds (mandates 15 and 16, `#five-seconds`). The node's
resident bridge pairs its links, keeps them paired and pairs them again, with or without a client
(design/bridge-and-ipc.md). What remains open is listed under Requirements and Risks.

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
| bridge (link) | each link heard within five seconds or concluded; failing the programs on a lost link; pairing again on the peer's next hello; the mesh's map | `rdma/mesh-link.c`, `rdma/mesh-program.c`, `rdma/mesh-verbs.h` |
| collectives library (sessions) | no clock of its own: its waits end at the bridge's `~0`, its bridge gone, a refused program, a GPU wait's bound or a revoke; a failed session closed and the next opened | `rdma/nccl-mesh.c`, `rdma/NCCL.md` |
| rank | its crossings' GPU waits bounded; calls delivered to it, and cancelled, by call delivery | metal-microbench `mesh_rank.m`, `rank.swift`, `docs/elastic.md#call-delivery` |
| driver (calls) | a call is a request to resident members, on the bridges' map taken at its start; a cancel is a request, and a requester gone cancels | metal-microbench `tools/mesh/grid.py`, `docs/elastic.md` |
| port (physical) | a fabric port whose cable link is up with no RDMA peer: bounced with backoff, logged | `bin/mesh-fabric-init.sh` (every keeper pass) |

## Mandates

Verbatim operator statements outrank everything else (metal-microbench AGENTS.md). The ones that bear on what is built
are first.

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
15. **Rule zero** (2026-10-18, after the M5's kernel panic from RDMA driver misuse): "rule zero of low level
   programming is 'You Aren't Allowed To Fuck Up Shared State Used By Other Services', however that is enforced or
   guaranteed in the particular time-sharing metaphor the host system uses."
16. **Five seconds** (2026-10-18): "you cannot be using a low level kernel like a dma or networking send/rcv channel for
   more than 5 seconds without something happening/concluding"; "no matter what apple says, i say 5 seconds. this is
   not unfair, and it is an easy standard." (`#five-seconds`)

**The tensions, resolved.** Mandates 2 and 3 place recovery between calls; mandate 1 asks that segmentation never
affect aggregate performance. A lost link fails the executions on it (mandate 13), and aggregate performance across a
segmentation is the next execution's to keep, its setup kept short by the resident bridge (no bridges or probes per
call). Mandate 4 counts a live peer declared lost as the worst failure, and mandate 16 bounds every use of a channel at
five seconds: a conclusion ends this end's use of a channel and declares nothing about the peer, whose next hello pairs
the link again (`#five-seconds`).

## Requirements

Each with its source and the mechanism that meets it (design/bridge-and-ipc.md, rdma/mesh.h, rdma/mesh-plan.h,
rdma/nccl.h); **open** where none is decided. Which of them source holds, and what has run, is git's and output_data's
to say (AGENTS.md: assignments expire), not this table's.

| id | requirement | source | mechanism |
|---|---|---|---|
| Q1 | A drop of a direct link costs the executions on it, which fail as values; the next pairs on the link again | mandates 1, 11, 13 | a loss fails each flow on the link (`#loss`); the port pairs again on the peer's next hello once its quiet window ends (`#pause`, `#five-seconds` B9) |
| Q2 | A long drop costs no more than the work the remaining members could not do | mandates 1, 4 | open: the survivors' next executions run on the map without the lost node |
| Q3 | Recovery between executions only; none inside one | mandates 2, 3, 13 | a failed flow is never resumed; the next program is a new flow on the map as it is then (design/bridge-and-ipc.md#flows) |
| Q4 | A link down when a program starts fails the program at once; the link pairs again on its own | mandates 1, 4 | `#first` (EHOSTUNREACH, ENOLCK); the port's hellos (`#pause`) |
| Q5 | Peer identity independent of the link: the node, its bridge's boot, and the pairing's index | mandate 3 | the hello carries node and boot; the port table counts pairings (design/bridge-and-ipc.md#ports) |
| Q6 | Loss is observed at the completion queue, the control socket and the link's frames; the data path keeps no timer | mandates 3, 8, 16 | a failed completion stops its link (`#loss`); the event loop alone reads the five-second conclusions (`#five-seconds` B2, B4), never a progress thread |
| Q7 | A bounded loop pairs again while the bridge is up, whether or not a client is attached | mandates 3, 14 | every port's turn, for the bridge's life: hellos every 3 s while it has no peer (`#pause`); no neighbour warm-up |
| Q8 | A realized program survives link loss without being realized again | mandate 3 | dropped by mandate 13: the execution fails and is run again |
| Q9 | A port move needs no configuration | mandates 3, 5, 11 | the bridge finds each port's peer itself (design/bridge-and-ipc.md#ports) |
| Q10 | A partition of more than two nodes is a topology change, not an error storm | mandates 1, 6 | each side's executions across the cut fail once, naming the node (`#loss`); each side's next runs on its map (`#the-drivers-side`) |
| Q11 | A cable pull and replug, three times, with time to repair and to the first completed execution recorded | mandates 1, 3, 10 | a measurement on hardware, its record in output_data |
| Q12 | Recovery costs nothing on the healthy path | mandate 7 | the progress threads read no recovery state but the link's `served`, once a pass; the event loop alone reads the five-second conclusions (`#five-seconds`); an idle link carries one keepalive frame a second and a record every 3 s |
| Q13 | No per-transfer TCP; TCP at pairing only | mandate 8 | the control socket carries HELLO, LINK, RECORD and FAREWELL (design/bridge-and-ipc.md#protocol), nothing per transfer or per program |
| Q14 | Every wait has a guaranteed end, and no DMA or network channel is used five seconds without something happening on it or its scope concluding | mandates 12, 15, 16; metal-microbench output_data/structural-20261018/WAITS.md, rule-zero/RULE-ZERO.md | `#five-seconds`, scope by scope. The bridge's loop waits on kevent64 with a 250 ms tick that only paces its turns; a port's progress threads park on their port's bell at most 1 ms at a time while anything is theirs to look for (3 s otherwise), and on its raised guard (design/bridge-and-ipc.md#progress); a control socket never blocks (a message queues behind what the socket has not taken: mesh-link.c port_say, port_flush); a device's verbs calls are made only on its port's own queue, on hearing the peer, at a link event or past a conclusion's quiet window; a program makes none and waits on no window; the stop waits on no queue, thread or verbs call (design/bridge-and-ipc.md#stop) |
| Q15 | A reconnection keeps context, PD and registrations; no verbs object is released while the bridge runs but a link's lanes at its next pairing | RDMA-KERNEL-RECOVERY.md "October 18" | a link off or a detached interface makes no verbs call; a concluded link's lanes are moved to the error state once its quiet window ends, where its interface stayed up (`#five-seconds` B6 (b)); its next pairing, on hearing the peer, moves them to the error state, destroys them and makes them again (d6dceac's sequence: `pairs_down`, `pair_up`, LINK, RTR, RTS), the completion queues, context, PD and registration kept, a queue pair whose destruction fails kept for the next pairing (B16); a device that fails with its link active is forgotten with no verbs call and opened afresh; the kernel releases what was forgotten at the bridge's exit (design/bridge-and-ipc.md#lanes) |
| Q16 | No cross-queue GPU spin; every spun-on word in a canceller's set | memory no_cross_queue_gpu_spins | a failed flow cancels its client's words toward the node (prepared-machine.md M12); every GPU wait is bounded besides (`#five-seconds` W3a) |
| Q19 | A link that answers its peer's hello but not RDMA does not flap at call level | mandate 1 | open: such a link hears no keepalive frame and is concluded five seconds after its last (`#five-seconds` B2), its programs failing as values, and it pairs again at the peer's next hello, so it flaps at that pace |
| Q20 | Algorithms total over every membership and topology (ring, star, uncabled pairs) | mandate 6 | the map is every algorithm's operand (mesh-plan.h `struct mesh_link_map`); the bridges forward nothing, so members no cable joins are joined through the nodes between, members of the communicator with no contribution (metal-microbench output_data/structural-20261018/DESIGN.md decision 1; mesh-plan.h `mesh_compile`'s `contributors`); open: libnccl-mesh's sessions bind a channel to every member, which needs a cable to each (rdma/NCCL.md "Routes between unlinked ranks") |

## The module

### five-seconds

The operator, 2026-10-18:

> rule zero of low level programming is 'You Aren't Allowed To Fuck Up Shared State Used By Other Services', however
> that is enforced or guaranteed in the particular time-sharing metaphor the host system uses.

> the platform deadline is 5 seconds ... you cannot be using a low level kernel like a dma or networking send/rcv
> channel for more than 5 seconds without something happening/concluding. you do not need any more information than
> this; this only affects extremely specific namespaces or program execution scopes and it should be very tidy and fast
> to deduce which they are.

> no matter what apple says, i say 5 seconds. this is not unfair, and it is an easy standard.

One constant carries it: mesh.h `MESH_PLATFORM_NS`, 5 × 10⁹ ns, the operator's number. Every bound below is that
constant or a fixed fraction of it. Remedies are taken in this order: prevention (never enter the call or the state),
structure (hold nothing shared while waiting), and a bound of at most five seconds that concludes the scope (a value
failed, a lane's requests flushed, a link concluded, a GPU wait ended). No remedy sends SIGKILL to a verbs owner. A
conclusion ends this end's use of a channel and declares nothing about the peer: its next hello pairs the link again.
Outside these scopes time only measures or paces an emission.

**The scopes, and what concludes each** (metal-microbench output_data/structural-20261018/rule-zero/RULE-ZERO.md §1;
the labels are its mechanisms', which the code cites):

| scope | what holds shared state | what concludes it within five seconds |
|---|---|---|
| S1 a verbs call that enters the driver | the driver's work-loop gate and, through it, IOKit's termination thread | calls made only on the port's own queue, under its guard; none inside a quiet window (B9) or a terminating peer interface's (B9b), none of a lost pairing (B10), none while any port's call is out past the bound (B12); the devices read again only for a new Thunderbolt interface (B11); a call out past the bound concludes its port around it (B12) |
| S2 a posted work request | the controller's queues, and the window rows its SGE names | a SEND awaiting its receive: its lane's gate retiring nothing for the bound fails its flow (B4); a receive awaiting its data: concluded with its link (B2) |
| S3 the link's standing receives | as S2 | the keepalive's frame whenever the link is up (B1), heard, or the link concluded (B2) |
| S4 the control socket and the link | a network channel shared with configd and the driver | something heard within the bound: a whole message, or the keepalive's frames moving, the record retold every 3 s (B3); else the link concluded (B2) |
| S5 a GPU or host wait on a landing | the GPU's queue (a spin inside a kernel) | every GPU wait bounded by its stream's polls, at most `NCCL_MESH_POLLS` (W3a); a host wait ends at `~0`, which B2, B4 and the stop store, or at its bridge gone (W3c) |
| S6 a stopped program's posted receives, and every lane of a concluded link | as S2, into rows that may be released to other clients | no lane leased while a flow may have requests on it (B5); the link concluded and paired again, its lanes moved to the error state and destroyed (B6); rows held while a request may land (B7); no post on a lane no longer served (B8) |
| S7 the stop and the exit's reclaim | IOKit's termination thread, through `deallocPD` | the stop takes every catchable fatal signal (B15) and waits on nothing; the exit is held to the quiet windows' end, at most the bound past the stop's start (B9, B9b) |

Host threads waiting on a word, a futex or a socket hold no driver, controller or GPU state; they are no scope, and
each ends as a value once the scope it depends on concludes.

**The bridge's mechanisms** (rdma/mesh-link.c, mesh-program.c, mesh-verbs.h):

| id | mechanism | where |
|---|---|---|
| B1 | The link's keepalive frame on lane 0 whenever the link is up, at both ends: every `keepalive_ns` while a flow of the port is listed (the configuration's `keepalive_us`), else every `MESH_PLATFORM_NS`/5 (1 s), an idle beat that counts as no movement, so the progress threads still park | `link_alive` |
| B2 | A port whose control socket heard nothing for the bound (stamped at the socket's making, at each whole message and as the keepalive's frames heard move), whose link is up and heard no keepalive frame for the bound since this pairing's first, or whose frame stayed in flight as long, is concluded: `port_lost(ETIMEDOUT)`, no verbs call, its programs' words `~0`, its socket closed so the peer reads its end. Read on the event loop alone; fbc53b37's `heard_at`, at the operator's bound | `port_silent` |
| B3 | This node's record retold at its current sequence on every paired port each liveness turn (`MESH_PAUSE_NS`, 3 s); a receiver drops a record no newer than its newest, so the map and its serials stay. B2 needs B3: a record every 10 s against a 5 s bound loses a healthy link (link_model.out `control-oscillate`) | `record_retell` |
| B4 | A served program lane whose gate holds frames posted and not retired, none retired for the bound, fails its holder as a value (ETIMEDOUT), never the link. A ring posts each record once the one behind it lands, so a SEND held that long is one its peer's flow does not post for; between calls a gate is empty | `gates_turn` |
| B5 | A flow's lanes' lease words become owner\|DIRTY as the flow is made, and its stop clears DIRTY only on a lane it left with nothing posted, so no word reads free, or another program's, while a stopped program's requests are posted on its lane | `lanes_take`, `lanes_release` |
| B6 | (a) A flow's stop that left a lane with requests posted concludes its link (no verbs call at the stop), so both ends pair it afresh. (b) A concluded link's lanes moved to the error state once its quiet window ends, where its interface stayed up. (c) Every lane at the link's next pairing, before its destruction, so no queue pair is destroyed with receives posted | `lanes_release`, `link_turn` (`broken`), `link_job` (`LINK_JOB_FLUSH`), `pairs_down`, `pairs_error` |
| B7 | A flow that left a lane with requests posted is kept with its rows (its client's, where retired) until the link's lanes' generation moves (the next pairing destroyed the queue pairs) | `flow_stop`, `flows_turn`, `clients_turn` |
| B8 | No post on a lane the link no longer serves: a flow is served only while every lane it rides is in `served`, which a loss clears before the socket closes; a FAREWELL heard loses the link at once; the first failed completion or poll on any lane stops the whole link | `port_pass`, `link_lost`, `link_failed`, `message_heard` |
| B9 | The quiet window: each conclusion holds its port's jobs (an open, a make, a connect, a put-off move) for the bound, and the exit until the latest window, so no driver call and no exit falls inside the termination a conclusion may follow; the devices are read again only after every window | `port_lost`, `quiet_until`, `windows_end` |
| B9b | A Thunderbolt RDMA peer interface's termination, heard in its first phase (`kIOTerminatedNotification`; rule-zero/EXIT-RECLAIM.md P2): every port's window and the exit's held the bound from then, and the port its `RID` names halted and concluded with no flush; at the start, IOKit's registry waited quiet, at most the bound, before the devices are read (P3, as a window) | `peer_terminated`, `terminations_turn`, `main` |
| B10 | A job of a lost pairing makes no call: its pairing is compared as it starts and between its steps, and the peer's numbers are copied at its dispatch | `link_job`, `link_lapsed` |
| B11 | The devices are read again only for an `en<N>` interface that attached or came on with no port: awdl, llw, utun and bridge interfaces never enter the driver | `network_heard` |
| B12 | A port whose open, link job or flow stop's move is out past the bound, or whose progress threads did not hold for a settle or a guarded change within it (the change not made), is concluded around it once: `port_lost(ETIMEDOUT)`, its programs' words `~0` from the event loop. While any port's call is so out, no port dispatches a driver call and the devices are not read. The call itself cannot be ended; the port opens again once it returns and its threads have told the conclusion's epoch (22891630^ mesh-flight.h's pending-operation age, per port) | `ports_stalled`, `port_wait` |
| B13 | The progress threads park on a raised guard, and a settle's or a guard's raiser parks (at most the bound), so a guarded call in the driver costs no core | `port_progress`, `port_wait` |
| B15 | No stop signal is lost: every catchable signal whose default ends the process takes the stop, its kqueue filter added before its default is ignored | `main` |
| B16 | Convergence: a failed connect or make leaves the port open, a hello heard on a port not open opens it, and a queue pair whose destruction fails keeps its handle for the next pairing (4df30806 `down_pair`), so a live context is never forgotten | `hello_heard`, `link_turn`, `pairs_down` |
| P4 | The provider's doorbell fault absorbed (rule-zero/EXIT-RECLAIM.md §2.4): a SIGSEGV or SIGBUS whose pc lies in libthunderboltrdma's text, on a thread that posts on a port's queue pairs, maps one anonymous page at the faulting page by hint (never `MAP_FIXED`, so nothing mapped is replaced), and the port is halted and then lost as at its link's going off (ENETDOWN, no verbs call). The bridge does not exit. Any other fault takes the default action | `fault_heard`, `terminations_turn` |

**The clients' mechanisms:**

| id | mechanism | where |
|---|---|---|
| C1 | A program's start waits on nothing: it hands the program to the bridge, and its first wait is its first landing | mesh-call.c `mesh_transfers_start` |
| C2 | A numerical executor's worker leaves at `running` 0, its reference passing to the calls it submitted, so no thread waits on a call | mesh-call.c `mesh_worker_settle` |
| C5 | A wait on the map's serial ends at its change, its bound, or its bridge's exit (its pid 0, another, or gone) | mesh-dataflow.c `mesh_map_wait` |
| W3a | Every GPU wait bounded: a stream's polls, 0 or more than `NCCL_MESH_POLLS` being `NCCL_MESH_POLLS`, the bound over the slowest load period read (rdma/NCCL.md "Bounded waits"); a wait at its bound fails its session | nccl.h, nccl-mesh-metal.m, mesh.metal |
| W3b | A lease word let go by compare-exchange on the word observed, so a DIRTY its bridge set stands | mesh-call.c `mesh_lanes_release` |
| W3c | The library's progress thread reads its bridge's pid and boot: a bridge gone (a fault stores no `~0`) cancels its session's waits | nccl-mesh.c `bridge_gone`, `progress_run` |
| W3d | The members' agreement compares the cables each end leased, so ends that would stripe over different cables fail as values | nccl-mesh.c `session_agree` |
| N5 | The library keeps no silent-link clock: B2, B4 and W3a conclude what one would | nccl-mesh.c `progress_run` |
| W7a | The rank's crossings and the Neural Engine partner's waits take the bounded form | metal-microbench `mesh_rank.m`, `neural_engine.swift` |
| F8 | A stream's waits end at its last landing, at `~0`, or the bound past its last landing with nothing landed | mesh-stream.c |
| — | NCCL's bootstrap: the second gather, which holds a rank's lane claims, and every later bootstrap send or receive are bounded by the bound; the first gather carries no DMA and enters no driver, and a revoke or the caller's process ends it | nccl-mesh.c `bootstrap_bound` |

**What no user-space mechanism ends** (RULE-ZERO.md §0 and §5):

- A thread already inside the driver: an `IOConnectCall` into IORDMAFamilyUC waits uninterruptibly (`THREAD_UNINT`),
  and no signal ends it, SIGKILL included. B12 concludes its port around it and makes no further driver call, but the
  thread holds the process's exit.
- The kernel's reclaim at the exit: it runs inline on the exiting thread and reaches `deallocPD`. A termination longer
  than five seconds that overlaps an exit is the 2026-09-05 and 2026-10-18 deadlock, the one stuck path the model
  leaves with every mechanism above (link_model.out `power-fixed`); under the operator's rule it is the platform's
  violation.
- An uncatchable end (SIGKILL by hand, jetsam) during a termination, and a termination that begins during the reclaim
  itself (about 1 ms).
- A post and the driver's queue shutdown are two steps; P4 makes a fault of a post in that gap survivable.
- A forgotten context's user client may hold its device's termination open (RULE-ZERO.md §5, 3).
- Unknowns: the M5's GPU load period (W3a's bound passes five seconds there if it is slower than 1.1 us); whether a SEND
  held on one queue pair blocks another's frames (B4 against B2); whether a move to the error state can block during a
  termination (B6 waits for the windows).

**Cost.** Pairing again waits up to five seconds after every conclusion, clean ones included (B9), and every
segmentation adds it to its recovery (mandate 1).

### detection

A loss is a link-off or detach of the port's own interface (M24), its peer interface terminating (`#five-seconds` B9b),
the control socket ending (a FAREWELL before its end makes it an orderly one; a reset; the kernel's own end of a
connection whose peer is gone), a send on it failing, a failed completion or poll on any lane of the link, a fault of
the provider's doorbell (P4), or five seconds with nothing heard on the link (B2). A post refused is its flow's alone,
and so is a SEND held five seconds (B4). No clock decides the peer's membership: a conclusion ends this end's use
of the link, and the peer's next hello pairs it again. A session's waits end at the `~0` each loss stores (rdma/NCCL.md
"Failures").

### loss

A link's loss fails every flow on it: its client's pending words toward the node `~0` and its status the loss's code
(ECONNRESET where the peer said farewell, ETIMEDOUT where five seconds concluded it, ENETDOWN where its interface went
off); a flow holds no queue pair, so none is destroyed. The link's lanes (design/bridge-and-ipc.md#lanes) are served no
more before its socket closes and are left as they are, no verbs call made; once the port's quiet window ends they are
moved to the error state where the link was connected and its interface stayed up (`#five-seconds` B6 (b)), and the
link's next pairing, on hearing the peer, moves them to the error state, destroys them and makes them again with the
sequence d6dceac paired by, and reads the port's own address again (its attributes and GID, which the LINK it tells
names): a peer's restart can move the port's GID table, and the address of the port's opening failed every lane's
connection at the peer with EIO, the link pairing again every six seconds (2026-10-19, after the Mini's power loss).
A job of the lost pairing still on the port's queue makes no call (B10).

The first failed completion or poll on any lane stops the whole link: every lane served no more, each one's program
failed, and the link concluded at once, both ends pairing it afresh. A post refused, or a descriptor outside the
window, fails its flow alone. A program's stop frees a lane it left with nothing posted at once, and concludes the
link where it left one with requests posted (B6 (a)): a communicator kept across calls leaves nothing posted until its
destroy.

A link job that fails (making or connecting) concludes the link, which pairs again on hearing the peer (B16); a refused
registration leaves the port down until its link next comes on (`#registration`). A loss of the port itself (link off,
interface detached) keeps the port's device: its context, protection domain and registrations are used again at the
next pairing; a device that fails with its link active (its context gone with the peer's power) is forgotten with no
verbs call and opened afresh, the kernel releasing the old one at the bridge's exit. A flow's own failure is told to
no one: the peer's program ends by what ends it at its own end (its link's loss, a SEND held five seconds, its
caller's revoke).

### farewell

A bridge that stops queues FAREWELL on every link, carrying why it stops (the signal it took, or its binary changed),
and closes the socket; the kernel sends what it holds. A FAREWELL heard is the peer's "release everything": the link is
lost at once and nothing is posted toward the peer again, its programs fail with ECONNRESET, the link is logged
"parted" with the reason (never "lost"), and it pairs again when the bridge is back.

### diagnosis

The operator, 2026-10-15, after the Mac mini restarted at 16:22 with nothing of ours running on it: "ensuring this class
of error is diagnosable from mesh logs (e.g. we can sync critical importance events over the rdma link s.t. a failing
node tells its neighbors along the mesh bridge *how* it is beefing it, if it is beefing it, even if it cannot
self-diagnose or self-correct beefing-it-ness)", and "a crash could spread contagiously across a mesh by a computer
shutting down for unknown reasons *during* a DMA operation, or cause kernel errors which persist across the lifespan
of affected peer processes". That restart's cause was in the Mini's power controller's record alone (a supply loss,
`por,por_vddrtc`), and logs without times could not place it.

- Every bridge log line carries the wall clock (UTC, ms) and the node's uptime (mesh-verbs.h `say`).
- Each node's record carries its boot (`kern.boottime`) and its available memory (`kern.memorystatus_level` of its
  memory), sampled at most 10 s apart (mesh-plan.h `struct mesh_health`); nothing else rides the map (the operator,
  2026-10-18: "almost no featue or responsibility could ever be assigned to the bridge service from thereon"). A change
  of these alone is written under the region's health serial (`hdr.health`), never the map serial: the map serial
  moves only where the mesh changes, and a communicator's session reads the map again where it moved and reopens only
  where the change changes its own routes or their links' costs (nccl-mesh.c `map_digest`, `session_ensure`).
- A survivor logs, at the moment of loss, the node it lost, or that parted with its reason, or that five seconds
  concluded; when the map drops a node no paired path reaches, that node and how long before it was heard; a node that
  joins or comes back, and how long it was gone (mesh-program.c `map_put`, `record_heard`).
- `mesh-stat` prints every node's record (`nodes`); node-info prints the node's `last_reset`, its power controller's
  newest record (`kern.shutdownreason`: `por,...` its supply was lost, `wdog,...` its watchdog reset it), read through
  its rank.
- A supply loss gives the node itself no warning: its last words are the record its neighbours hold and their logs,
  and the reason is read at its next boot.

### registration

A port's device registers the whole window when it comes up (design/prepared-machine.md#M09), checked against that
device's max_mr on the port's own queue. A registration the device refuses (the window's extents past its max_mr,
EFBIG, or an extent `ibv_reg_mr` refuses) is logged with the extent, its bytes, the errno and the extents the device
holds; the port keeps those extents and is down until its link next comes on, when it registers the rest; no clock
retries it. A node of several cables registers the window once a device (80 extents each at an 80 GiB window), which
has not yet run on more than one device of a node, so a refusal of the second device is the first thing to look for
when a Studio's second cable will not pair.

### pause

3 s (mesh.h `MESH_PAUSE_NS`) between a port's hellos while it has no peer, a paced emission that decides nothing; a
port whose paired link was lost says one at once, and a dial or greeting that failed keeps the pace, so nothing is
tried faster (mesh be9731b: "failed completions, queue-pair teardown and immediate re-pairing preceded the Sep 5 kernel
panic"). A conclusion's quiet window holds the port's next open, make or connect five seconds (`#five-seconds` B9). The
device opens at the start, at its link's coming on, and at a hello heard while it is not open (B16). The same 3 s is
the bridge's liveness turn: its record retold (B3), its binary and host name read, client slots whose process is gone
retired. No setting changes it.

### first

A program starts with no message to its peer: its flows are made as it starts, on the lanes its client leased at its
communicator's init, and its start waits on nothing (`#five-seconds` C1). It fails at once as a value, its status
naming the node, where the node is on no cable of this one (EHOSTUNREACH: the bridges forward nothing), where no cable
carries its lease (ENOLCK), where it is not a lane pair's shape (EINVAL) or where a port's table is full (E2BIG). A
peer whose program starts later finds its posted receives already waiting, and a SEND waits in the hardware for its
receive (TN3205's credit), so no member's start waits on another's; a SEND held five seconds fails its flow (B4).
Members whose programs of an execution are made from different maps find it in their session's first position (the
digest travels as data: nccl-mesh.c's agreement), not at the start.

### install

Every exit of a bridge is a reclaim that may meet a termination (`#five-seconds` S7), so a build stops nothing:
`make all` in rdma/ builds every image into `rdma/.build` and installs nothing, and `make mesh-c` installs nothing.
`make install` renames each built image whose bytes differ into rdma/, in `IMAGES`' order, the libraries first and the
bridge's binary last, and the binary anew wherever libmesh changed (the bridge watches its binary alone); a bridge whose
binary changed stops at once and launchd starts the new one (design/bridge-and-ipc.md#stop). The deploys that mean to
install run `make all install` (bin/mesh-user.sh, bin/mesh-converge.sh between calls once the node is quiet,
bin/mesh-application.py, rdma/torch-mesh/setup.py). A build that installed was a stop: the M5 took three so on
2026-10-18 (metal-microbench output_data/structural-20261018/rule-zero/SHARED-STATE.md F2).

### the driver's side

Each execution reads its node's bridge's map at its start (mesh-plan.h `mesh_map`; no probes, no bridges of its own);
a link the bridges pair again is in the next execution's map. Calls reach their members, and are cancelled, through
the members' resident ranks (metal-microbench docs/elastic.md#call-delivery), never through a bridge.

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
   Answered by `#pause` (3 s) and the quiet window (`#five-seconds` B9); the cause of the panic itself is not
   established.
2. **Neighbour cache after a replug.** RDMA-RULES.md: after any replug both sides fail RTR unless the neighbour cache
   is warmed. Not done; the hellos on the link may warm it; a pull may show otherwise.
3. **Arrival counts reset by their last arriver** (mb docs/kernels.md#decode-attention, 2026-10-08): the last-arrival
   merges leave a count nonzero when a dispatch is cut, and a later run of the same program merges wrongly without
   failing. No path here runs a program again after a cut: a failed execution's process prepares its programs again,
   their counts starting at zero.
4. **What only the pair can show.** A move to the error state and its flushed completions, the guard's park, P4's
   absorbed fault and B9b's notification have no reading on Apple's provider (RULE-ZERO.md §7 lists them, with the
   M5's GPU load period and the one-row time against 4.24 ms with B1's idle beat).
5. **The residuals of `#five-seconds`**, a termination longer than five seconds that overlaps an exit first.

## Failures by kind and topology

Topologies: **P** the TB5 pair; **R** a ring or complete mesh of 3-6 nodes; **U** uncabled pairs, joined through the
nodes between; **H** heterogeneous links (TB5, TB4, Ethernet); **Q** several communicators on one link (four lane
pairs).

| # | failure | what happens |
|---|---|---|
| F1 | a direct link down mid-call | the executions on it fail; the link pairs again at the peer's next hello once its quiet window ends |
| F3 | a link down when a program starts | the program fails at once (EHOSTUNREACH, or ENOLCK where no paired cable carries its lease) |
| F6 | the peer bridge stops or crashes | its FAREWELL, else its socket's end or five seconds unheard (B2): the executions fail; the link pairs again with the new bridge |
| F7 | the peer's program fails or ends | nothing is told: this side's SENDs toward it are held and fail their flow at five seconds (B4), its other waits end by that or its caller's revoke; a lane its stop left with requests posted concludes the link at that end |
| F8 | the peer node reboots or loses power | its peer interface's termination (B9b), the link-off, or five seconds unheard: the executions fail, no port job and no exit inside the termination's window, the port's device kept (opened afresh where its context died) |
| F9 | the control plane (LAN, ssh, mDNS) lost, the cable up | the bridges' links are unaffected; the driver's reach is |
| F10 | the data path dead, the control socket alive | no keepalive frame heard for five seconds concludes the link (B2); a SEND held five seconds fails its flow (B4) |
| F11 | one direction lost | the end that hears nothing concludes the link at five seconds (B2), and its socket's close ends it at the other |
| F12 | a link degrades (speed, errors) | the Balancer on evidence; an error completion stops the link (`#loss`) |
| F13 | one link of several drops | the executions on it fail; the others' links are untouched |
| F14 | a node between two members on a route drops | its links' loss fails the programs on them at both ends of each; the next execution's membership is the current map's |
| F15 | a cable moved to another port or peer | the old port's executions fail; the new port pairs with whatever node it reaches |
| F18 | the driver's own node fails | the stream stops (one driver) |
| F19 | a node of several cables stops in a mesh whose survivors hold a cycle | its last record ends at the first bridge that has heard it (design/bridge-and-ipc.md#records) |
| F20 | a lane's completion fails (an error status, a failed poll) | the whole link stops (`#loss`): every lane served no more, its programs failed, the link concluded at once and paired afresh. A post refused is its flow's alone |
| F21 | a device refuses the window's registration | logged with the extent; the port is down until its link next comes on (`#registration`) |
| F23 | members' programs of one execution made from maps a change apart | the session's first position carries the digest, which the members' agreement compares (`#first`) |
| F26 | a verbs call stuck in the driver | its port concluded around it at five seconds and no driver call from any port meanwhile; the port opens again once the call returns (B12) |
| F27 | the provider's doorbell faults (a post across the driver's queue shutdown) | absorbed: the port lost as at its link's going off; the bridge does not exit (P4) |

## The review this asks for

1. The device-state, pause and five-second rules checked with a pull and replug on hardware, once runtime is allowed.
2. Q2, Q19 and Q20's open part decided, or dropped; Q11 and F1-F15 run where they can be.
