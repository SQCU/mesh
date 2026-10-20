# The transport: one owner of the device, clients of its pages

This is the description of how data moves between nodes, who may touch what, how it fails and what that costs. It
answers, in terms of the objects the code holds, the operator's questions of 2026-10-19. Where it and older text in
this repository disagree, this text is the description and the older text is history.

Incident records it cites: [RDMA-KERNEL-RECOVERY.md](RDMA-KERNEL-RECOVERY.md) (September 5, October 16-18). Apple's
transport note: [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt).

## 1. The objects

- **The registered window.** One POSIX shared-memory object per node, `/mesh0`, mapped by the bridge and registered
  with every Thunderbolt RDMA device of the node by that port's agent as it opens the device (section 14), as memory
  regions of 1 GiB extents
  aligned to 1 GiB (`tbt_post_recv` keeps only the low 32 bits of an SGE address with the region's key, so a region
  crossing a 4 GiB boundary aliased receives 4 GiB low: September 5). Every address a request names is inside it.
- **The page table.** The window's rows (one row is one block, 16 KiB on the pair) and, per row, an owner word and the
  arena page it maps (`mesh_buffer`, `mesh_page_entry`), claimed by compare-and-swap on bitmaps. A client's memory is
  rows whose owner word is its own. "Lease" in the code means only this: rows owned. It has no term, renewal or expiry.
- **A link.** The two ports at the ends of one cable and the queue pairs the bridges make on them at each pairing:
  ten unreliable-connected queue pairs (index 0 carries the keepalive frame), created, moved through INIT, RTR and RTS,
  on two completion queues of the port's device context.
- **A posted receive.** An `ibv_recv_wr` with one SGE: an address in the window, a length, a key. The queue pair's
  next arriving SEND fills the oldest posted receive. Its completion is an entry in the receive completion queue.
- **A SEND.** An `ibv_send_wr` with one SGE naming a range of the window. TN3205: only SEND and RECV exist on this
  transport (no RDMA WRITE or READ), a SEND and the receive it fills must cover the same number of 4 KiB frames, and a
  SEND is held by the hardware's credit until the peer has a receive posted.
- **A readiness word.** A 64-bit word in shared memory with exactly one writer and one or more readers: a client's
  "supplied" word for a position it sends, the bridge's "landed" word for a position it received. All-ones
  (~0) in a landed word is a position whose data never came (section 10).

Nothing else is needed to move data: rows, requests naming rows, completions, and readiness words.

## 2. Who owns what

**The bridge** (`rdma/mesh-flow`: a resident parent per node, started by launchd, and an agent per port that the
parent spawns, section 14) owns the device contexts, the protection domains, the registration, every queue pair and
completion queue, every request it posts, which rows a receive or a SEND names, and in what order a link's queue pairs
carry which call's positions. Its port agents are the only processes that hold a device's objects or post on them; the
parent reads the device list alone.

**A client** (a rank, a tensor function, torch) owns its rows, and for each of its calls the positions it sends and
receives. It reads and writes its own rows and its readiness words. It names nothing else.

Why the cut is there: the device's state lives as long as the node and its cables (registration of most of RAM, links
paired and re-paired while no call runs, ten queue pairs per link shared by every client), and a call lives for one
call and may fail, be revoked or be killed. A call that held device state would leave it half-done at its end, and
another party would have to repair it with verbs calls. No call holds any.

**Why some of the bridge's state must be unnameable by clients.** Four kinds of state are shared by parties that fail
independently, have invariants no access can check in time, and fail irreversibly. For them, privacy is the only
enforcement (complete mediation: Anderson's reference monitor, 1972; Saltzer and Schroeder, 1975; capabilities, Dennis
and Van Horn, 1966):

1. *Memory the device writes.* A DMA write cannot be undone and the NIC checks nothing beyond the region's key. A
   client that can form an address in the window can write where another client's receive lands, or keep using rows
   after they are given away. Clients hold their own rows only.
2. *Device registers and state machines.* Queue-pair handles, requests, SGEs, doorbells and the RESET-INIT-RTR-RTS
   sequence have preconditions the hardware does not check, or checks by faulting: a post while another queue pair of
   the device is destroyed stored to a doorbell mapping that had moved (October 16-17). One agent sequences them.
3. *Finite shared resources.* Ten queue pairs a link, each queue's frames and its share of the completion queue.
   Deadlock and overcommit come from parties taking parts of what they need on their own (dining philosophers;
   Coffman's hold-and-wait); the fixes, an arbiter or one acquisition order, need the allocation state out of reach.
   `ibv_post_send` accepts posts past the queue's frames and corrupts later, so its credit has one writer.
4. *State that outlives a client.* Posted receives, SENDs in flight, which landing belongs to whom. Progress of others
   must not depend on a party that can vanish; only a single owner can always complete what it started.

The only words both sides touch are single-writer readiness words: the client writes "supplied", the bridge writes
"landed".

## 3. There is no protocol, except a link's pairing

A protocol is needed only where one side must learn something neither side can know beforehand. Here that is the
queue-pair numbers, packet sequence numbers and GIDs the hardware assigns at each pairing (TN3205's out-of-band
endpoint exchange): HELLO and LINK on the control socket, once a pairing. RECORD carries each node's links and memory
(the map), which is data, not a dialogue.

Everything about a call is known before it runs: both ends compile the same schedule, so each position's length and
order are known at both ends. Data goes SEND into a posted receive. A side that supplies nothing is replaced by a dud
of the scheduled length (section 10). A lost link is a driver event (link off, an error completion, the control
socket's end). There are no requests, acknowledgements, handshakes, agreements, digests, sessions, meetings or stamps.

## 4. SEND and RECV do not fail

In correct code:
- `ibv_post_send` and `ibv_post_recv` fail only on the caller's error: past the queue's capacity (prevented by
  counting frames) or a bad SGE or key (prevented by construction).
- An error completion appears only when a queue pair is moved to the error state (a flush) or on a local protection
  error. The bridge never moves a queue pair to the error state while it is in use.
- On an unreliable-connected queue pair a SEND does not fail because of the remote end; under TN3205's credit it
  waits until a receive is posted there.

What does happen: nothing arrives (the other side supplied nothing, or the cable is gone), and the driver's device
events (link off, the peer interface's termination). The M5's bridge log from 2026-10-16 00:08Z to 2026-10-19 holds
no failed completion at all. It holds 176 link losses: 100 the bridge concluded itself over receives a stopped
communicator had left posted (section 12), the rest its own conclusions at restarts, peers' stops and interfaces going
down, and connects failing with EIO from its own stale addressing after the Mini's power loss (2026-10-19).

## 5. Every loop of ours can stop; only a kernel call cannot

Every loop the transport needs polls memory without blocking:
- the bridge's receive thread (`ibv_poll_cq`, which returns at once) and send thread (reads supplied words);
- a host wait and a GPU wait on a landed word.

Any of them stops when a flag says so: the bridge's stop, a watchdog or heartbeat process's semaphore, the deadline
below, a link-off event. A stopped receive loop leaves its receives posted, which harms nothing: the rows stay the
bridge's, and what lands later is recorded as late.

The operator's deadline (2026-10-18, verbatim): "the platform deadline is 5 seconds. i'm telling this to you right now.
block quote me. you cannot be using a low level kernel like a dma or networking send/rcv channel for more than 5
seconds without something happening/concluding."

The only thing that cannot be stopped is a thread inside a kernel call that waits on something that never happens. A
process in that state (STAT `U`) ignores every signal, SIGKILL included. Observed:
- **September 5 (the Mini).** Test processes stayed in `U` in the release of their verbs objects, holding their
  protection domains and queue pairs. `ibv_alloc_pd` then failed for every process, all three ports went down with
  cables attached, `ibv_devinfo` hung, `shutdown -r now` never completed, and the machine needed a power cycle. Some
  had been sent SIGKILL. The kill did not make the wait: a process's exit releases its verbs objects whatever ended
  it, SIGTERM and a return from main included, and a release that waits on a terminating device waits the same way.
- **September 5 and October 18 (the M5).** `ibv_dealloc_pd`, called while a peer's device was terminating, entered
  `deallocPD`, which holds the provider's work-loop gate while it waits for the device's termination, whose thread needs
  that gate. On October 18 this held IOKit's termination for 17 hours, and the Wi-Fi/Bluetooth manager panicked the M5
  after waiting 65 s for its own termination. Every later open of the device blocked behind it.

So the bridge makes no call that can wait on a termination: no `ibv_dealloc_pd`, `ibv_dereg_mr`, `ibv_destroy_qp`,
`ibv_destroy_cq`, `ibv_close_device`, no drain. A device or queue pair it no longer uses is forgotten; the kernel
releases it at the process's exit.

Every process of ours can be ended by any signal at any time, SIGKILL included. A guard that keeps a process from
being killed is a second failure added to compensate for the first: it presumes the deadlock will happen, it does not
prevent it (an exit by SIGTERM or by main's return makes the same release), and it leaves a process nobody can stop.
The first failure is removed where it is: our code makes no release call, and nothing of ours waits forever, so a
kill always finds a process that can end. The one exposure left is the kernel's own release at an exit that coincides
with a device's termination (section 9).

## 6. Who looped forever

| date | what spun or blocked | whose |
|---|---|---|
| 2026-09-05 | test processes in the release of their verbs objects (state U) | the provider's release, entered by our processes' exits during the fault |
| 2026-09-05, 10-18 | `deallocPD` waiting on a termination that waits on its gate | the provider's deadlock, entered by our teardown call |
| 2026-10-02 | a GPU kernel spinning with no bound on a word only another command queue wrote; the M5 crashed | ours |
| 2026-10-17 | `pair_flush` spinning for `ib_drain_qp` markers the provider accepts and never completes | ours: waiting on something never promised |
| to 2026-10-19 | unbounded host and GPU waits on landed words, waits on peers in call setup | ours |

The data path of Apple's provider (`post_send`, `post_recv`, `poll_cq`) always returned. Every infinite wait on record
was ours, and the one kernel hang was entered through a call of ours.

Open: the Mini's two kernel data aborts at fault address 0x98, seconds after its bridge stopped, October 16 and 17
(each stop then destroyed queue pairs while posts could be in flight; trigger not established).

## 7. With one second of warning

The case: someone presses the Mac mini's power button within a second of a cable being pulled, or the cable goes
first.

There is no escape from a deadlock once it has formed: the thread is in `U`, the gate is held, and only the kernel's
own watchdog (the M5's 65 s panic) or a restart ends it. The second of warning cannot undo it; it can only make sure
nothing starts it. Since the bridge makes no call that waits on a termination (section 5), nothing does:

1. **The node going down** (the power button: macOS sends SIGTERM to launchd's agents). The bridge sets its stop flag,
   its two threads a port see it at their next pass and post nothing more, it sends FAREWELL on every paired port, and
   it exits. It makes no verbs call. Whether the Mini finishes shutting down within the second does not matter: if
   power goes first, its kernel state goes with it, and a machine that is off cannot hang.
2. **The node staying up** (the M5). It reads FAREWELL, or the link-off event, or the peer interface's termination.
   It stops serving that link and makes no verbs call on it. The termination finishes because nothing of the bridge
   holds the provider's gate. Calls with positions on that link get duds (section 10) and fail as values; every other
   port, process and service is untouched.

## 8. A crushed cable

No warning at all: the hardware is gone before any event says so. This is tenable on three conditions:

- **No teardown.** The surviving bridge makes no call that waits on the termination (section 5), so the termination
  completes and the node stays responsive. On 2026-10-19 the Mini lost power with the M5 running a bridge that makes
  no teardown call, and the M5 stayed up; the 2026-10-18 hang had a teardown call in it.
- **The port left, the others kept.** Each Thunderbolt port is a separate device context in a process of its own. The
  bridge stops serving the crushed port and makes no call on its handles; once the port's quiet window has passed its
  agent ends, and the next agent pairs the link again when the cable is back (section 14). The map drops the link, and
  the next calls plan over the remaining cables (two thirds, five sixths).
- **A post racing the loss.** A post issued in the instant the hardware went may touch a doorbell mapping that no
  longer exists. Recorded only with a queue pair being destroyed by our own call at the same time (October 16-17);
  with no destroy calls it has not been seen. Where it happens, the post's thread faults, and the port is lost, as
  the cable is.

Positions in flight on the crushed link are transport errors (section 11). Nothing needs administration.

## 9. Should the bridge ever stop on its own?

No. It does not stop because of a peer, a link, a client, a call, a time or a fault. A stop exists only to replace
its binary, and it is started from outside (the upgrade's SIGTERM). The stop is: a flag; the port threads see it and
post nothing more; FAREWELL on every paired port; exit. It makes no verbs call, needs no interrupt handler (the signal
handler only sets the flag), and cannot enter the October 18 deadlock, which needs a deallocation call in flight.

The one exposure left is the kernel's own release of the device at the process's exit coinciding with a termination,
which is what upgrades avoid by their timing: one node at a time, with no program running and the peer's bridge
steady.

Nothing of the node's brings a fabric port down: a port whose cable link is up with no RDMA peer is logged and held
(`bin/mesh-fabric-init.sh`), since a port taken down under the bridge's open device terminates the device beneath its
objects. A job an operator disables (`launchctl disable`) stays off: the keeper and the installers load no disabled job
(`bin/mesh-agent.sh`).

A fault the bridge's start or its pairing meets would recur at every boot, since launchd and the keeper start the
bridge at boot. The bridge therefore keeps a boot ledger beside its configuration (`boots`): the boot it started in,
and how many boots in a row ended while a bridge was in its first 10 minutes with the ledger set, a panic, a power loss
or a reset. At n of them, it opens no device for 60·2ⁿ⁻¹ s, at most an hour, and a stop signal ends it meanwhile. Its
stop clears the ledger, and so do 10 minutes of running. Such a fault then recurs at most once an hour, its report is
written at each boot, and the node stays reachable by ssh in between.

## 10. Duds

If a client has not supplied a position it scheduled by the position's deadline (the call's own, at most the
platform's 5 s), the bridge sends a dud: the scheduled number of frames of all-ones. The bridge chooses which rows
every SEND names, so a dud is a SEND whose SGE names the bridge's own all-ones rows (one run as long as the longest
position, written once at its start): nothing is copied into the client's rows or anywhere else. The receiving
bridge's receive fills as for any SEND, so every posted receive is consumed in order and nothing is left posted.

All-ones reads as failed wherever it lands: it is NaN in fp16 (0xFFFF) and fp32 (0xFFFFFFFF), and ~0 in a landed
word. A dud is a client error, recorded against the client and its call, never a transport error.

Positions of a call whose client failed, was revoked or exited are sent as duds at once, without waiting for a
deadline. Received positions whose consumer is gone land in their rows, and the rows are released as they land.

## 11. Reliability: three nines

The operator (2026-10-19, verbatim): "i am now demanding that there be 3 nines of reliability between bridge or
transport level errors. you need to transmit 1000 frames which are *normal* and have no errors in their transmission
for every 1 *error* wrt the bridge application. sending an all-1 dud buffer is not a transport error, but a client
error".

- **A frame** is one 4 KiB frame of a SEND the bridge posted for a scheduled position.
- **A normal frame** arrives intact in the receive the schedule gave it.
- **A transport error** is a frame that does not: lost, cut short, landed in another position's receive, held by the
  bridge past its deadline, or cancelled because a link was lost or concluded. Every frame in flight on a link when it
  is lost counts, and so does every frame scheduled on it until the calls on it end.
- **Not transport errors:** duds (a client's), and positions of calls that ended before they were sent.
- **The requirement:** at least 1,000 normal frames per transport error, counted per bridge.

What it rules out: any link loss the bridge causes itself. One link loss in a decode call at the pair's rate (on the
order of ten thousand frames a second) costs every frame scheduled until the call ends; a re-pair of about 8 s
costs tens of thousands, which then requires tens of millions of clean frames to stay at three nines. Before this
document the bridge concluded a link at every rank renewal and at every failed call (section 12), which no workload
can meet. After it, only physical events (a cable pulled or crushed, a node losing power) lose links.

Counted by the bridge, per port: frames posted, frames landed, duds sent, and transport-error frames by cause. A
summary of the four is part of each node's record.

## 12. The code before this document

What HEAD did against the sections above, by section:

- 2: every client mapped the whole window read-write; client rows held `struct mesh_send` (a prepared `ibv_send_wr`
  and the queue-pair pointer), which the bridge posted as it found it; clients named the addresses of the words the
  bridge writes (`mesh_delivery`); clients claimed queue pairs by compare-and-swap on `hdr->lanes`.
- 3: a per-communicator "session" (opened at the first crossing group, its rings in the communicator's rows, a digest
  agreement at its first position), the rank's "meets" before and after preparation, filler stamps and their checks.
- 4, 5: receives were posted ahead into a communicator's own rows and restarted for each communicator, so every
  communicator's end left receives posted ("dirty"); the bridge's only way to clear them was to conclude the whole
  link and pair it again (`lanes_release`, `link_turn`), ending every other communicator on the cable and the rank's
  call-delivery group with them, about 8 s each time. Communicators were kept from call to call to avoid that, and
  a failed member's end still did it (2026-10-19: one call's failed program build cost the next two calls).
- 9: the bridge stopped itself when its binary changed.
- 10: no duds; a member that supplied nothing made its peers wait to the bound of their GPU waits.
- 2 and 11, in metal-microbench: per-rank byte ceilings computed by the planner (`programs.py plans`, the engine's
  `withinPlan`) refused allocations the window could hold.

## 13. Reviewing transport code

The bridge is reviewed by design/rubrics.md, rubric M; libnccl-mesh and the schedule compiler by rubric C; the
programs that use them by rubric D.

## 14. The data path, decided by rubric M and C

The forms the rewrite chose between, ranked by design/rubrics.md, and the form the code takes.

The algorithm analysis these rows rest on is metal-microbench `output_data/bridge-20261019/FLOWS.md` section 7.

| Question | Ranked | Rubric |
|---|---|---|
| What a client names | a port and a lane pair (`bind[port][pair]`, 127) ≺ a node | M5, D7 |
| Who takes a queue pair | a client's compare-and-swap on `hdr->lanes` (HEAD before 127) ≺ the bridge binds it at the client's asking (127) ≺ no client takes one: the bridge stripes a node's stream over the link's lane pairs | M5, M2, M9 |
| How a client holds the stream | a request and an answer in its slot (127) ≺ its owner word taken while the stream is free, as rows are | M2 |
| Where receives go | into a communicator's rows, restarted per communicator (HEAD before 127) ≺ into the stream's in ring, posted by the bridge for the link's life | M4, M9 |
| How a holding ends | the link concluded ≺ every unit it declared or supplied sent, the unsupplied ones as duds, and one more | M4, M10 |
| Where the next holding starts | the bridge's answer in the client's slot (127) ≺ the stream's `next`, published once free, exchanged at init through NCCL's bootstrap | M2, C7 |
| A queue pair's re-pairing | moved to RESET (127, panicked the M5) ≺ destroyed and made again ≺ none: made and connected once in the bridge's life, a lost link down until a person relaunched it (129) ≺ none in a process: made and connected once in a port agent's life, released by its exit, the link paired again by the port's next agent (130) | M3, M4, M6 |
| A progress thread with nothing to do | parked on a bell clients ring (127) ≺ spinning while the link serves, waiting on the guard word while it serves nothing | M8 |
| The binary changing | the bridge stops itself ≺ it stops on the installer's SIGTERM alone | M3 |

**The stream.**
- Each link carries `MESH_LINK_STREAMS` (4) streams toward its peer node, each on a lane pair of its own, since a
  queue pair carries one ring (a SEND fills the oldest receive posted on it). Each stream has two classes: large units
  (the window's block) and small units (one frame). Each class is a ring of units in the bridge's arena, with a ready
  word and a landed word per unit (`struct mesh_stream`).
- A stream's units of a class ride its lane of that class in order, alike at both ends (`MESH_STREAM_PAIRS`, the pairs
  a stream stripes over, is one).
- Each stream has an identity, `wire`, which both bridges compute alike from its first lane's two ends (each port's
  GID and queue pair number) and which no other stream between the two nodes has.
- The receive thread posts receives into the in ring for the link's life, and raises a unit's landed word only once it
  and every unit before it have landed. A run has landed once its last unit's word says so, whichever lanes it rode.
- Clients see the stream's rings and words, its `peer` node, `wire`, `live`, `free` and `next`. They never see a port,
  lane, queue pair, request or completion.

**Holding it.**
- A process holds a stream as it holds rows: it takes the owner word from 0 to its own while `free` reads 1
  (`mesh_stream_take`), and gives it back to 0 when its last communicator on it ends (`mesh_stream_give`). The bridge
  answers nothing; it lets go the word of a client whose process is gone (`streams_let_go`).
- The holder's first units are the stream's `next`, which its communicator's init tells its peers through NCCL's
  bootstrap. One communicator holds a stream at a time. Each channel's head credit (a position published once the
  peer's position LAG back is consumed) sizes the ring for that channel alone, so two communicators on one ring could
  write a slot before it is read. Communicators that share a node pair hold different streams of its link: an init's
  first gather carries each rank's free streams (`mesh_streams_free`), and both ends of each pair of members take the
  stream of the least `wire` free at both (`mesh_stream_take`), so the two ends of a communicator hold the two ends of
  one stream. An init is refused (EBUSY) only where no stream toward a member's node is free at both ends.
- The holder declares how far it will supply each class (`declared`), with a `deadline` (uptime ns) that a live holder
  leaves at UINT64_MAX: no unit of a producer still at work is ever sent as a dud (design/rubrics.md C5 and Quantities,
  the operator 10-20). A holder that can supply no more, its group failed or revoked, sets the deadline to now, and
  the bridge sends its unsupplied declared units as duds at once. A consumer's own wait stays bounded where it waits.

**Duds, and a holding's end.**
- The send thread posts unit u when its ready word reads `1 + u / units`. Where u is declared, not supplied, and the
  deadline has passed, it posts a dud: a SEND from the bridge's all-ones block.
- When the owner word changes, the send thread ends the holding: every unit up to the most declared or supplied goes,
  the unsupplied ones as duds, and one unit more. Once every lane has sent them, the stream is free again, its `next`
  the first unit after them.

**A process a port** (version 130, its streams new at every pairing from 131; the operator's choice of 10-20).
- The bridge is a parent and an agent a port. The parent, launchd's job, holds the region, the map and records, the
  clients' slots, device discovery, hellos and the control sockets. A port's agent is `mesh-flow` spawned again by the
  parent: the same image (its LC_UUID checked, so a binary replaced on disk is no agent of a running parent), in a
  process group of its own, its parent's socket at descriptor 3. It opens the port's device, registers the window,
  makes and connects the link's ten queue pairs and runs the port's progress threads. The parent asks it to open,
  make, connect, halt and end; it answers each, naming the pairing it was asked for, and tells failed lanes, absorbed
  faults, guards not held and terminations it heard. Queue pairs exist only in an agent, and an agent is the bridge
  ("the bridge and only the bridge may ever have access to that exact data structure", 10-20).
- A link's ten queue pairs are made once and connected once in an agent's life. None is moved to another state,
  destroyed or forgotten; the kernel releases them at the agent's exit. A device holds ten usable queue pairs and a
  link takes all ten, so a returning peer's lanes cannot exist beside the old ones, and the old agent's exit is their
  one release (RESET panicked the M5 on 10-19; destroy is ruled out by its history, M3).
- A link lost once its agent was asked for lanes (a farewell, the control socket's end, a failed completion, the link
  going off, a termination, an absorbed fault, a protocol error, the agent's own end): its streams cancelled (landed
  words ~0), its lanes served no more (the agent halted, before the control socket closes), the port closed. Once the
  port's quiet window and any termination's have ended and no call is stuck (B9, B9b, B12), the parent asks the agent
  to end, reaps it and spawns the next, which opens the port and pairs with the peer whenever it answers. Nobody acts.
  Other ports' agents and links, the clients and their leases are untouched.
- An agent that ends unbidden loses its link the same way and is spawned again once the port's quiet window has
  passed; the k-th such end in a row within 60 s of its spawn, k from 2, holds the port's next spawn 60·2^(k−2) s, at
  most an hour (section 9's rule, a port's: "restarted, not crashing again of course"). An agent whose parent is gone halts its
  lanes and exits MESH_PLATFORM_NS later, its own windows held; a port's agent is spawned only while no other agent of
  its device lives (one of the launch before, whose parent ended unbidden), so a device's queue pairs are one agent's at
  a time and no make meets the device's ceiling. At the bridge's stop every agent is halted before the
  farewell, asked to end after the exit's hold, and awaited up to MESH_PLATFORM_NS.
- Streams across agents. A port's rings and words are the parent's, leased once. A new agent opens every stream new as
  its link connects: owner word 0, ready and landed words 0, `next` 0, free and live. A holding of the lost pairing
  ended with it: its holder's give (a compare-and-swap from its own client) then matches nothing, and its communicator
  reads its holding gone (mesh-call.h `mesh_stream_held`: live, its owner word its client's, its identity the one it
  took) and starts no group on it (libnccl-mesh `comm_failure`, `ncclCommGetAsyncError`). No holder keeps a stream from
  the link's next pairing by letting go or by never letting go (design/rubrics.md M4, 10-20). Nothing of the lost
  pairing is sent on the new one: that pairing's groups ended at their next look at a landed word ~0, the connect comes
  `MESH_PLATFORM_NS` and an agent's exit after the loss, and the two ends of a communicator take only an identity free
  at both (`wire`, a digest of this pairing's queue pair numbers and addresses).
- A LINK names the address its lanes were made on (a make reads the port's address again, as a peer's restart can move
  its GID table); the connect and each stream's identity take the peer's address from its LINK.
- Calls in flight on a lost link fail as values (ncclRemoteError) and the next call takes a stream of the new pairing:
  "a bridge crashing on a computer two hops down should result in the bridge being restarted, not crashing again of
  course, and becoming available again as a resource for collectives with nothing but an error channel indicating that
  calls-in-flight failed and ought to be retried" (the operator, 10-20).
- XMAGIC carries MESH_VERSION, so bridges of different versions never pair (129, 130, 131), and a client refuses a region of another version: both ends of a link, and every client, run the rule.
