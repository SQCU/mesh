# The transport: one owner of the device, clients of its pages

This is the description of how data moves between nodes, who may touch what, how it fails and what that costs. It
answers, in terms of the objects the code holds, the operator's questions of 2026-10-19. Where it and older text in
this repository disagree, this text is the description and the older text is history.

Incident records it cites: [RDMA-KERNEL-RECOVERY.md](RDMA-KERNEL-RECOVERY.md) (September 5, October 16-18). Apple's
transport note: [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt).

## 1. The objects

- **The registered window.** One POSIX shared-memory object per node, `/mesh0`, mapped by the bridge and registered
  with every Thunderbolt RDMA device of the node once, at the bridge's start, as memory regions of 1 GiB extents
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

**The bridge** (`rdma/mesh-flow`, one process per node, started by launchd and resident) owns the device contexts,
the protection domains, the registration, every queue pair and completion queue, every request it posts, which rows a
receive or a SEND names, and in what order a link's queue pairs carry which call's positions. It is the only process
that makes a verbs call.

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
- **The port forgotten, the others kept.** Each Thunderbolt port is a separate device context. The bridge stops
  serving the crushed port, keeps its handles unused, and reopens a fresh context if the port ever comes back. The map
  drops the link, and the next calls plan over the remaining cables (two thirds, five sixths).
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
