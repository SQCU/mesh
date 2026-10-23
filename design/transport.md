# The transport: one owner of the device, clients of its pages

This is the description of how data moves between nodes, who may touch what, how it fails and what that costs. It
answers, in terms of the objects the code holds, the operator's questions of 2026-10-19. Where it and older text in
this repository disagree, this text is the description and the older text is history.

Incident records it cites: [RDMA-KERNEL-RECOVERY.md](RDMA-KERNEL-RECOVERY.md) (September 5, October 16-18). Apple's
transport note: [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt).

## 1. The objects

- **The registered region.** One POSIX shared-memory object per node, `/mesh0`, mapped by the bridge and registered
  with every Thunderbolt RDMA device of the node by that port's child as it opens the device (section 14), one ibv_mr
  per extent, aligned so that each lies inside one 4 GiB bank. An extent is 2^k bytes, the least power of two from the
  arena's page up to the bank at which the region's extents number no more than the device's `max_mr` (`device_up`):
  1 GiB for the Studios' 86 GiB under max_mr 100. The facts are the region's size, `max_mr` and the bank;
  `max_mr_size` is not one (the devices report 16,384,000 and register 1 GiB extents). The bank's fact: `tbt_post_recv` keeps only the low 32 bits of an SGE address with the extent's key, so an
  extent crossing a 4 GiB boundary aliased receives 4 GiB low (September 5). Every address a request names is inside
  it.
- **The page table.** The region's rows (one row is one block, 16 KiB on the pair) and, per row, an owner word and the
  arena page it maps (`mesh_buffer`, `mesh_page_entry`), claimed by compare-and-swap on bitmaps. A client's memory is
  rows whose owner word is its own. "Lease" in the code means only this: rows owned. It has no term, renewal or expiry.
- **A link.** The two ports at the ends of one cable and the queue pairs the bridges make on them at each pairing:
  the unreliable-connected queue pairs of its plan, ten where both ends' devices hold ten (macOS 26's) and fewer where
  either holds fewer (the M5 Ultra's macOS 27 devices report max_qp 3, of which max_qp − 1, two, are usable:
  [HETEROGENEOUS-LINKS.md](HETEROGENEOUS-LINKS.md)),
  its streams' lane pairs and, where the plan has them, lanes 0 and 1 (nothing posted on either)
  ([HETEROGENEOUS-LINKS.md](HETEROGENEOUS-LINKS.md) section 10);
  created, moved through INIT, RTR and RTS, on two
  completion queues of the port's device context.
- **A posted receive.** An `ibv_recv_wr` with one SGE: an address in the region, a length, a key. The queue pair's
  next arriving SEND fills the oldest posted receive. Its completion is an entry in the receive completion queue.
- **A SEND.** An `ibv_send_wr` with one SGE naming a range of the region. TN3205: only SEND and RECV exist on this
  transport (no RDMA WRITE or READ), a SEND and the receive it fills cover the same number of 4 KiB frames, and a
  SEND is held by the hardware's credit until the peer has a receive posted.
- **A readiness word.** A 64-bit count in shared memory with exactly one writer and one or more readers, in pages: a
  holding's `ready` (the holder's: the pages it supplied) and its `sent` and `landed` (the bridge's: the pages whose
  SENDs completed, the pages landed), each in its list's order (section 14). All-ones (~0) in `sent` or `landed` is a
  lost link; a page whose data never came lands as a dud (section 10).

Nothing else is needed to move data: rows, requests naming rows, completions, and readiness words.

## 2. Who owns what

**The bridge** (`rdma/mesh-flow`: a resident parent per node, opened by its heartbeat, and a child per port that the
parent spawns, section 14) owns the device contexts, the protection domains, the registration, every queue pair and
completion queue, every request it posts, which rows a receive or a SEND names, and in what order a link's queue pairs
carry which call's positions. Its port children are the only processes that hold a device's objects or post on them; the
parent reads the device list alone.

**The region outlives a restart of its bridge** (layout version MESH_LAYOUT, apart from the wire's MESH_VERSION). A
bridge stopped by SIGHUP (its heartbeat's restart, a deploy) or ended by a crash leaves its region as it is. The next
bridge adopts the region, the same memory, where its layout is that bridge's:
- clients' slots, rows and leases stand, and the GPU buffers over them stay valid;
- the port children register the same pages again, so nothing is wired twice and nothing is reloaded;
- the bridge's own pages (each port's link pages, its streams' lists and time rings, the duds' block, the map) are
  recorded in the region and taken again;
- every stream is set not live (so no take succeeds), its landed words ~0, so a communicator of the old pairing fails
  and is made again; its owner word is left as found, and the port's next child stores 0 there as it opens the stream
  new (`streams_open`). That child is spawned only after the earlier launch's child of its device has exited
  (`child_spawn`, EVFILT_PROC), so a reader of that 0 reads after every post of the old child. A store of 0 by the
  adopting bridge would order nothing against them (`mesh_pinned`, section 14);
- a retirement a dead process left half done is finished.

A second bridge of the node ends at its port's bind, before it reads the region, so a region a live bridge serves is
not touched.

**A release frees it** (design/rubrics.md "Failure across interfaces" 6). Every stop of the bridge keeps its region; a
release is the heartbeat's: it stops the bridge by SIGTERM (its children end, so the kernel frees every object of their
devices), and once the bridge has ended it makes the region *expire*, its layout version made 0. A bridge that finds a region of another layout expires it the same way and ends
(EX_TEMPFAIL). No party signals a client. An expired region is:
- attached by no new client (mesh_open attaches a region of its own layout alone, and an expired one's is 0);
- returned to each client that holds a lease or communicator of it as a value, at its next lease or communicator
  (ncclRemoteError, and ncclMeshRegionExpired: nccl.h Memory); a client the library made no Metal buffer for frees what
  it holds of it, or ends, and the library then unmaps it and closes its descriptor; a client with Metal buffers over
  it (its GPU work and residency set may hold those pages after they are released) ends; what a client still holds
  stays mapped and valid until it frees it or ends;
- unlinked by the heartbeat once none of its holders lives, and its memory is then the system's. Its holders are
  recorded in the region, each by its pid and its process's start (so a pid the system gives again is no holder): the
  bridge (`bridge_pid` and `bridge_start`), each child a bridge spawned (the children register the region and hold it
  until their exit, which can outlast their bridge's), and each client slot's process (its owner's pid and `start`).
  The header's fields the heartbeat reads (the magic, the version, the offsets, `clients` and `bridge_pid`), the slots
  and the bridge's area keep their places in every layout, so the heartbeat reads the holders of a region of any
  layout.

No second region is made beside a holder of the first, because on the Mini (24 GB, a 17 GB region) GPU work on
one region beside another's registration wired all its memory and its watchdog halted the node (2026-10-20): the
heartbeat opens a bridge only once no expired region is named. A process that maps the region without a slot (a reader,
or a child forked by a client without exec) is not recorded; none registers it or makes it resident.

**The heartbeat** (rdma/mesh-heartbeat, the job launchd keeps alive, `io.mesh.bridge`) supervises the bridge. It holds no
device or socket of the mesh; it maps the region only for a moment at a time, to read its holders and its bridge's
progress and to write its expiry and the hang statistic (section 16). It opens the bridge (a child in a process group of its own, so the bridge serves
on if the heartbeat ends), closes it on the operator's signal and hears its end as a kernel event (EVFILT_PROC
NOTE_EXIT), not inferred from a delay. It is the one process that unlinks an expired region, and the watchdog of its
bridge: it ends a bridge whose event loop has not turned for MESH_PLATFORM_NS (section 16). Its signals, sent with
`launchctl kill SIG user/$(id -u)/io.mesh.bridge`:
- SIGHUP opens the bridge, or restarts it: the bridge closes by SIGHUP (its region kept), the heartbeat runs its own
  image again (a deploy's new binaries both, bin/mesh-user.sh), and the bridge opens and adopts the region;
- SIGUSR1 releases: the bridge closes by SIGTERM, the region is unlinked once none of its holders lives, and no bridge
  opens until SIGHUP;
- SIGTERM and SIGINT (launchd's stop of the job, a shutdown) release and end the heartbeat, which exits 0, and launchd's
  KeepAlive {SuccessfulExit: false} does not start it again.

A bridge that ends unasked opens again 60 s later, so the mesh is restarted at most once a minute (rubrics 6: "do
not stop meshes hundreds of times per second, minute, or hour"); nothing else is built for a crash (rubrics position
5). The operator, 2026-10-21: "launchd heartbeat reopen code is required by the user by block quote and will never be
overridden implicitly as a side effect of any other message or statement". A bridge that ended finding its region
expired opens again once the heartbeat has unlinked that region. A job whose
definition predates the heartbeat runs the bridge directly until the job's next load (launchd keeps a loaded
definition): its stops keep the region as ever, and it has no release.

**A client** (a rank, a tensor function, torch) owns its rows, and for each of its calls the positions it sends and
receives. It reads and writes its own rows and its readiness words. It names nothing else.

Why the cut is there: the device's state lives as long as the node and its cables (registration of most of RAM, links
paired and re-paired while no call runs, a link's queue pairs shared by every client), and a call lives for one
call and may fail, be revoked or be killed. A call that held device state would leave it half-done at its end, and
another party would have to repair it with verbs calls. No call holds any.

**Why some of the bridge's state is unnameable by clients.** Four kinds of state are shared by parties that fail
independently, have invariants no access can check in time, and fail irreversibly. For them, privacy is the only
enforcement (complete mediation: Anderson's reference monitor, 1972; Saltzer and Schroeder, 1975; capabilities, Dennis
and Van Horn, 1966):

1. *Memory the device writes.* A DMA write cannot be undone and the NIC checks nothing beyond the extent's key. A
   client that can form an address in the region can write where another client's receive lands, or keep using rows
   after they are given away. Clients hold their own rows only.
2. *Device registers and state machines.* Queue-pair handles, requests, SGEs, doorbells and the RESET-INIT-RTR-RTS
   sequence have preconditions the hardware does not check, or checks by faulting: a post while another queue pair of
   the device is destroyed stored to a doorbell mapping that had moved (October 16-17). One child sequences them.
3. *Finite shared resources.* A link's queue pairs, each queue's frames and its share of the completion queue.
   Deadlock and overcommit come from parties taking parts of what they need on their own (dining philosophers;
   Coffman's hold-and-wait); the fixes, an arbiter or one acquisition order, need the allocation state out of reach.
   `ibv_post_send` accepts posts past the queue's frames and corrupts later, so its credit has one writer.
4. *State that outlives a client.* Posted receives, SENDs in flight, which landing belongs to whom. Others' progress
   cannot hang on a party that can vanish. A single owner finishes what it started.

The only words both sides touch are single-writer readiness words: the client writes "supplied", the bridge writes
"landed".

## 3. There is no protocol, except a link's pairing

A protocol exists here only where one side learns something neither side knows beforehand. Here that is the
queue-pair numbers, packet sequence numbers and GIDs the hardware assigns at each pairing (TN3205's out-of-band
endpoint exchange): HELLO and LINK on the control socket, once a pairing. RECORD carries each node's links and memory
(the map), which is data, not a dialogue.

Everything about a call is known before it runs: both ends compile the same schedule, so each position's length and
order are known at both ends. Data goes SEND into a posted receive. A side that supplies nothing is replaced by a dud,
an all-ones page (section 10). A lost link is a driver event (link off, an error completion, the control
socket's end). There are no requests, acknowledgements, handshakes, agreements, digests, sessions, meetings or stamps.

## 4. SEND and RECV do not fail

In correct code:
- `ibv_post_send` and `ibv_post_recv` fail only on the caller's error: past the queue's capacity (prevented by
  counting frames) or a bad SGE or key (prevented by construction).
- An error completion appears only when a queue pair is moved to the error state (a flush) or on a local protection
  error. The bridge moves no queue pair to the error state.
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

Any of them stops when a flag says so: the bridge's stop, a watchdog or heartbeat process's semaphore, a revoke (a
host wait), its poll bound (a GPU wait), a link-off event. A stopped receive loop leaves its receives posted, which
harms nothing: the rows stay the bridge's, and what lands later is recorded as late.

The operator's deadline (2026-10-18, verbatim): "the platform deadline is 5 seconds. i'm telling this to you right now.
block quote me. you cannot be using a low level kernel like a dma or networking send/rcv channel for more than 5
seconds without something happening/concluding."

Since 2026-10-22 no wait of the bridge or the collective on a peer's data is bounded by a clock, and the bridge decides
nothing by one (section 14: "we don't introduce race conditions or use timing anywhere"). The deadline bounds the
bridge's own lifecycle (a driver call's job, a quiet window, a child's end) and the bootstrap's phases. A group's
presentation waits for its bridge to take its entries (`taken`), and a host-path group's wait for its own GPU work, and
a destroy's, end at the command buffer's completion status, which the platform's watchdog ends with an error where the
buffer overruns (a buffer not yet committed fails a group at once, as a value, and a destroy does not await it:
nothing runs it while its committer waits). A group's waits re-read the revoke and its holding between looks (rubric
M3, 10-18), and no transaction fails because time passed (2026-10-22). None of these waits holds the library's groups'
lock: each releases it for the wait and takes it again after, as a condition variable's wait releases its mutex, so
other communicators' groups and revokes go on (rubric M4). A GPU wait keeps its poll bound; a link
carries no keepalive.

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

Section 16 gives each its form, and the watchdog for what no form excludes.

## 7. With one second of warning

The case: someone presses the Mac mini's power button within a second of a cable being pulled, or the cable goes
first.

There is no escape from a deadlock once it has formed: the thread is in `U`, the gate is held, and only the kernel's
own watchdog (the M5's 65 s panic) or a restart ends it. The second of warning cannot undo it; it can only make sure
nothing starts it. Since the bridge makes no call that waits on a termination (section 5), nothing does:

1. **The node going down** (the power button: macOS sends SIGTERM to launchd's agents). The bridge sets its stop flag,
   its two threads a port see it at their next pass and post nothing more, and it exits; its control sockets end with
   it. It makes no verbs call. Whether the Mini finishes shutting down within the second does not matter: if
   power goes first, its kernel state goes with it, and a machine that is off cannot hang.
2. **The node staying up** (the M5). It reads the control socket's end, or the link-off event, or the peer interface's
   termination.
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
  child ends, and the next child pairs the link again when the cable is back (section 14). The map drops the link, and
  the next calls plan over the remaining cables (two thirds, five sixths).
- **A post racing the loss.** A post issued in the instant the hardware went may touch a doorbell mapping that no
  longer exists. Recorded only with a queue pair being destroyed by our own call at the same time (October 16-17);
  with no destroy calls it has not been seen. Where it happens, the provider's fault is absorbed in the port's child
  (section 15, "Terminations and faults"), and the parent loses the port as at its link's going off, as the cable is.
  The child ends once the port's quiet window and the termination's have passed, so its exit, and the kernel's release
  of its device's objects with it, does not fall inside the termination (section 5).

Positions in flight on the crushed link are transport errors (section 11). Nothing needs administration.

## 9. When the bridge stops

It does not stop because of a peer, a link, a client, a call, a time or a fault. A stop exists only to replace
its binary, or to release the node's memory, and it is started from outside (its heartbeat's SIGHUP, or its SIGTERM:
section 2). Its one end that is not a stop is its heartbeat's SIGKILL where its own event loop has not turned for the
platform deadline (section 16), which the next bridge meets as a crash: the region adopted, the reopen a minute later. The stop is: a flag; the port threads see it and
post nothing more; exit. It makes no verbs call, needs no interrupt handler (the signal
handler only sets the flag), and cannot enter the October 18 deadlock, which needs a deallocation call in flight.

The one exposure left is the kernel's own release of the device at the process's exit coinciding with a termination,
which is what upgrades avoid by their timing: one node at a time, with no program running and the peer's bridge
steady.

Nothing of the node's brings a fabric port down: a port whose cable link is up with no RDMA peer is logged and held
(`bin/mesh-fabric-init.sh`), since a port taken down under the bridge's open device terminates the device beneath its
objects. A job an operator disables (`launchctl disable`) stays off: the installers load no disabled job
(`bin/mesh-launchd-agent.sh`).

A bridge that ends unasked opens again a minute later (its heartbeat, section 2), and a port's child that ends unbidden
is held as section 14 says; nothing else is kept across a boot.

## 10. Duds

A dud is the whole page of all-ones. The bridge chooses which page every SEND names, so a dud is a SEND whose SGE
names a page of the bridge's own all-ones block (written once at its start): nothing is copied into the client's rows
or anywhere else. The receiving bridge's receive fills as for any SEND.

The bridge sends a dud only for a page of an entry whose rows lie outside the page table or hold no page (a holder's
error). It sends none of its own: none at a deadline (a holding has none) and none for a holding that ended (the
operator, 2026-10-22: "the bridge has as little code as possible and does as little as possible").

All-ones reads as failed wherever it lands: it is NaN in fp16 (0xFFFF) and fp32 (0xFFFFFFFF), and ~0 in a landed
word. A dud is a client error, recorded against the client and its call, not a transport error.

Pages received for a holder that is gone land in its rows: a row a receive posted and not landed names stays held until
the receive lands or its port's child exits (section 14, "A holding's end").

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

**Whose loss it is** (rubric M10: "the form whose accounts assign each lost frame to its real cause and owner"). Each
loss is booked, in the links table's `code` and the log, by the event that told it, which names its owner:
- the control socket's end (ENETRESET, or the socket's own error): the peer node's bridge ended;
- the link's going off or its interface's detach (ENETDOWN), the provider's fault absorbed in the port's child
  (ENETDOWN): this port's device or cable;
- the peer interface's termination (ENXIO): the device at the cable's far end gone;
- lanes failed (a failed completion or post's error), no stream live (LINK_STREAMS_ENDED): this link's lanes;
- a job or a child past the platform deadline (ETIMEDOUT, section 16): this bridge's own conclusion;
- a child ended unbidden (ECHILD), a job that failed (its errno, ECANCELED, EPIPE), a message this end cannot pair
  with (EPROTO), no room in the region (ENOMEM): this end, by what failed.

A peer bridge's restart, its release and its crash all reach this node as its control socket's end, and are booked
alike: the owner is the same (that node's bridge), and the reason is that node's own, written once where it is known,
in its heartbeat's log ("asked to close (SIGHUP...)", "ended unasked", "ended by signal"). Ranked: a FAREWELL message
carrying the reason to every peer (ad55cbf4^; M2's lower pole, "a second protocol on top of RDMA", and a second store
of a fact its node already keeps) ≺ the end of the socket the pairing made, joined with the peer's own log by time. Until
this change a termination was booked ETIMEDOUT, the code of the bridge's own conclusion at the deadline: a physical
event booked as the bridge's own (M10's lower pole, inverted) ≺ ENXIO.

## 12. The code before this document

What HEAD did against the sections above, by section:

- 2: every client mapped the whole region read-write; client rows held `struct mesh_send` (a prepared `ibv_send_wr`
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
  `withinPlan`) refused allocations the region could hold.

## 13. The rubrics

design/rubrics.md ranks the bridge's forms by rubric M, libnccl-mesh's and the schedule compiler's by rubric C, and
the programs' that use them by rubric D.

## 14. The data path, decided by rubric M and C

The forms the rewrite chose between, ranked by design/rubrics.md, and the form the code takes.

The algorithm analysis these rows rest on is metal-microbench `output_data/bridge-20261019/FLOWS.md` section 7.

| Question | Ranked | Rubric |
|---|---|---|
| What a client names | a port and a lane pair (`bind[port][pair]`, 127) ≺ a node | M5, D7 |
| Who takes a queue pair | a client's compare-and-swap on `hdr->lanes` (HEAD before 127) ≺ the bridge binds it at the client's asking (127) ≺ no client takes one: the bridge stripes a node's stream over the link's lane pairs | M5, M2, M9 |
| How a client holds the stream | a request and an answer in its slot (127) ≺ its owner word taken while the stream is free, as rows are | M2 |
| Where receives go | into a communicator's rows, restarted per communicator (HEAD before 127) ≺ into the stream's in ring, posted by the bridge for the link's life, which clients copy out of (127-131) ≺ into the holder's own landing pages as it presents them, the bridge's spare page between holdings (132) ≺ into the holder's own landing pages, nothing posted between holdings (133) | M4, M9, M8 |
| How a holding ends | the link concluded ≺ every unit it declared or supplied sent, the unsupplied ones as duds, and one more (132) ≺ at once at its give: nothing more of it posted, the stream free (133) ≺ at once at its give or its holder's retirement, each unit told before it is posted and posted only while the holding is its holder's: with nothing outstanding the stream free, else the stream not live for its pairing (134) | M4, M10 |
| Where the next holding starts | the bridge's answer in the client's slot (127) ≺ each lane's counts sampled as a holding begins, exchanged as `start` words and MAXed, spares and duds between (129-132) ≺ where the last one ended at each end, the stream's pages posted told as facts, the filler presented by the collective (133) ≺ where the last one ended, equal at both ends of a live stream by construction, the facts kept as a statistic (134) | M2, C7 |
| A holding that ends with units outstanding | its receives' owner's every row held until they land (`touching`), the next communicator's filler and its wait taking them up (133) ≺ the queue pairs moved to Error and every request flushed (the operator's permission of 10-22, not used) ≺ the stream not live for its pairing, a link with no live stream lost by the port child's exit, the window's rows held until each unit completes or the child exits (134) | M3, M4 |
| Which rows a reclaim keeps | every row of a holder a stream still names (`mesh_touched`, 133: D(P) taken as everything P held) ≺ W(t), the rows the units posted and not completed name, the plan's window in flight, each until its unit completes or its port's child exits (134; design/extents-and-liveness.md, d) | M4 |
| A queue pair's re-pairing | moved to RESET (127, panicked the M5) ≺ destroyed and made again ≺ none: made and connected once in the bridge's life, a lost link down until a person relaunched it (129) ≺ none in a process: made and connected once in a port child's life, released by its exit, the link paired again by the port's next child (130) | M3, M4, M6 |
| A progress thread with nothing to do | parked on a bell clients ring (127) ≺ spinning while the link serves, waiting on its lanes served while it serves nothing | M8 |
| The binary changing | the bridge stops itself ≺ it stops on its heartbeat's SIGHUP alone | M3 |

**The boundary (version 132; the operator, 10-21).** "the bridge, which is an api for controlling DMA, and the collective
clients and data processing code, which present buffers for transmission and accept whatever packet or frame size the
bridge chooses to use. the bridge chooses, and it chooses based upoin the actual driver and api the rdma firmware
supplies. only the bridge is allowed to know these configuration details." And: "zero copy methods must always be used
over access by copy because memory bandwidth and flops are too scarce on this hardware api." Versions 127 to 131 had
clients copy into the bridge's rings and count positions against them (a lag credit), so clients knew frames and units;
the fix of version 125's real defect (clients holding work requests and queue pairs) had moved the data itself into the
bridge. A client naming its own pages holds no DMA state, and needs no ring.

**The stream.**
- Each link carries up to `MESH_LINK_STREAMS` (4) streams toward its peer node, each on a lane pair of its own (as
  many streams as the link's plan has lane pairs for: [HETEROGENEOUS-LINKS.md](HETEROGENEOUS-LINKS.md)). Every unit on
  a stream's lanes is a whole page: page p of the stream's sequence over its pairing goes on its lane p mod 2, as unit
  p div 2 there, and each end posts the sequence in order.
- Each stream has an identity, `wire`, which both bridges compute alike from its first lane's two ends (each port's
  GID and queue pair number) and which no other stream between the two nodes has.
- A holding has two lists of entries the holder presents in order: a send list and a landing list. An entry is a
  range of the holder's rows (`struct mesh_entry {row, pages, word, base}`: `pages` rows from row `row`), each row a
  page, read in order through the page table: the range is the entry's block table (design/pages-and-functions.md).
  Each list is a sequence of pages, and page i of a send list lands as page i of the peer's landing list. Each end
  lays its own pages where its plan put them, a scattered buffer being several entries: scattered pages in one client
  may land as one run in another, and one buffer may go to several runs in several clients, with nothing copied. No
  entry names a byte offset, so none can lie off a page. Every placement is the holder's plan, fixed
  before the holding's first call (the operator, 2026-10-21: "all placements can be demanded as ahead of time known
  compiled plans which do not change dynamically during a chain of function calls"). Three page counts over the
  holding, each in its list's order with every page before: `ready` (the holder's: produced), `sent` and `landed` (the
  bridge's: SENDs completed; landed). A send entry's page k is ready once the word at `word` (0: the holding's
  `ready`) reads at least `base` + k + 1. A landing entry names no word, and the bridge reads none: its receives are
  posted as the plan presents them (what a landing writes, below). The holder sees no lane, unit, frame or queue pair.
- Every SEND and every receive is one whole page, the page its row names (the bridge reads the row's page as it posts
  it: one read a page). A send entry whose rows lie outside the page table, or a row that holds no page, goes as duds,
  and such a landing entry's pages land in the spare page (the port's link page). So each SEND fills its receive
  exactly (TN3205), ends whose lists differ garble their own pages without failing the link, and neither end knows the
  other's addresses.
- The send thread posts a unit from its page once the entry's word says the page is ready; the receive thread posts
  each receive straight into its page as the entry is presented (at most a window ahead), and raises `landed` once
  every page before has landed, whichever lane carried it. A SEND whose receive is not yet posted waits in the
  hardware's credit (TN3205): posting a receive is the credit, and nothing else paces a holding (no lag).
- What a landing writes. The operator, 2026-10-22: "data procesisng code gets to write their own guards in their own
  source. if source says to write continuously into ab uffer, the nic must write continuously into that buffer even
  if this means that dat aprocessing code can race or lose parts of a stream." And on 2026-10-21, that programs own
  their buffers' order: "keep only the landing wait, one acquire fence, a poll bound for the node". So:
  - the transport writes landings as the plan presents them;
  - a program that reuses a buffer guards that buffer in its own source, or races;
  - landings are not ordered after the stream's earlier work. Here the contract departs from NCCL's stream
    semantics, whose kernels copy out of NCCL's own buffers after the stream's earlier work.
- The words only the bridge writes: `live`, `sent`, `landed`, `taken`, `posts` (each of the stream's two lanes' units
  over its pairing each way, posted and completed), `rows` (the rings of the row each unit names, `MESH_LANE_UNITS` a
  lane each way) and the two time rings (`posted_at`: an entry's first SEND posted;
  `landed_at`: its last page landed; mach_absolute_time ticks read once in the pass, after its counts). The time
  rings are measurement and decide nothing.
- Lines (layout 139, the bridge audit of 2026-10-22). Each writer's words of a stream lie on a 128-byte line of their
  own (`hw.cachelinesize`; mesh.h `MESH_LINE`), so a store by one writer does not take a line from a reader of
  another's: the holder's `owner`, `sends` and `lands`; `ready`, the GPU's or the holder's; the send thread's `sent`
  and `taken[0]`; the receive thread's `taken[1]` and `landed`; and `posts`, a line each way. A lane's progress at the
  child is the same: the send thread's SENDs posted and done, and the receive thread's receives posted and landed, on
  lines of their own (`struct lane_state`).
- The count first. A completion pass stores the count a GPU waits on (`sent` or `landed`, with release) before
  anything else: the clock read, `posts`, and the time rings come after it, and the clock is read only where an entry
  is to be timed.

**Holding it** (version 133; the operator, 2026-10-22, on the holding's start of 129-132: "something is sampled
diachronically four times then we do a funny little guess ... this entire code path needs to be remove and rewritten
from first principles ... we never do tensor transports for dynamic or jit programs which cannot compile their data
dependencies or functional behavior. another is that we don't introduce race conditions or use timing anywhere. why?
because using timing is always slower than not using timing, and we require the lowest attainble overhead on all
transactions in bridge/collective"; "why is this a derived value instead of explicitly presented configuration which
is transduced simply from a configuration operand and several invariants?"; "this should be something that never
fails"; and "the collective layer gets to optimize data placement and movement, and finally the bridge has as little
code as possible and does as little as possible", which may tell consumers facts, "an error channel return kind of
statistic, not a data dependent control flow for a collective verb or a bridge transaction").
- Two invariants. (1) A stream's lanes carry its holders' entries' pages and nothing else: no keepalive, no spare
  receive between holdings, no dud of the bridge's own. (2) Both ends' lanes count from 0 when the port's child makes
  its queue pairs, and each end posts the stream's pages in the sequence's order. So a SEND's n-th unit on a lane
  fills the peer's n-th receive there (TN3205), and a page's place is a function of the two ends' entries alone.
  Posting in order keeps both lanes' counts a function of one number, the stream's pages posted: two lanes posted
  apart could leave a holding that ended at counts that no sequence of pages fills.
- A process holds a stream as it holds rows: it takes the owner word from 0 to its own by one compare-and-swap
  (`mesh_stream_take`), nothing awaited. The send thread then begins the holding, its walk at the stream's pages sent
  so far, and the receive thread its landing walk at the stream's pages posted so far: no unit of an earlier holding
  is outstanding then (a holding's end, below).
- The facts. The bridge tells each of a stream's two lanes' units over its pairing, each way, posted and completed
  (`posts`), and the row each unit names (`rows`). By (1) nothing is posted on a stream between holdings, so `posted`
  is a constant while no holder presents: the taker reads it as it takes the stream, and nothing is read while it moves.
- The difference, a statistic (version 134). Each rank puts its streams' pages posted in the report its
  communicator's init already gathers (NCCL's own init exchanges each connection's endpoint the same way: C1), and each
  channel keeps the two ends' difference (`differ`), which nothing reads to decide anything. On a live stream it is 0
  by construction: a holding that ends with nothing outstanding at both ends left every SEND completed into a posted
  receive (TN3205) and every receive landed from a SEND, and one that ends otherwise takes its stream out of the
  pairing. The filler of version 133 (dud SENDs and receives into the spare page presented first, entries of
  `MESH_NO_ROW`, and the init's wait for an earlier holding's receives to land) is deleted; the operator, 2026-10-22,
  on waiting for a future communicator's filler: "dispose of".
- One communicator holds a stream at a time; communicators that share a node pair hold different streams of its link,
  each end taking the stream of the least `wire` free at both. An init is refused (EBUSY) only where none is.
- A SEND is posted when its row's word (`ready`) passes it; a receive as its entry is presented, within its lane's
  window. Both are data, and neither has a deadline.

**A holding's end** (version 134; the operator, 2026-10-22: "if we have a graph serializaiton that we know tells us
that specific buffers used by a specific lcient have no future writes targeting them they can be freed earlier? ...
some algorithm tries to assume that all prorgams have arbitrary access patterns and communication schedules for the
sake of breadth, but in so doing relaxes the problem socpe they are working on so severely that they describe a less
stable or worse-converging algorithm"; on `mesh_touched` pinning every row of a dead client: "this is a claude subagent
excess and must be corrected"). The analysis, with its literature, is design/extents-and-liveness.md, in its terms:
D(P) the extents a program declares, U(P, t) those it uses, L(P, t) those some later event of it may still use, and
W(t) those requests already handed to the hardware reference, the plan's window in flight.
- The holder gives the stream by setting bit 63 of its owner word (`MESH_STREAM_GIVEN`). A holder gone gives nothing:
  its exit retires its client slot (the event loop's EVFILT_PROC: the death is that event, not a poll), and the send
  thread's read of the slot (`owned`) finds it not the holder's, as a give would; `mesh_pinned` reads a stream of a
  client gone as given. The send thread ends the
  holding at the first pass that reads the give, or its client's retirement: nothing more of it is posted, its entries
  taken and not posted are dropped, and the receive thread is told (`ended`). Each progress thread reads whether the
  holding is still its holder's (its owner word not given, its client's slot its own) once a pass, not once a unit
  (mesh-link.c `holding_kept`). Each unit is told before it is posted: its row in its lane's ring, then its lane's
  count, with release (`unit_told`); nothing is read back and nothing awaited. Until layout 139 each unit took a
  sequentially consistent store and loads, so the core waited for its store to be seen.
  The holding's end orders the rest: the receive thread stores `stopped` after its last post of the holding, and the
  send thread stores the owner word (`stream_free`) after reading it. So a reader that has read the owner word past the
  holding has read every unit ever posted for it (`mesh_pinned`, under Rows). Nothing in the bridge closes a holding
  with its peer: no message, no totals, no clock.
- Once the receive thread has stopped too, the send thread reads the stream's two lanes. Where every SEND posted has
  completed and every receive posted has landed, the holding's words are made new and the owner word goes to 0: the
  stream is free.
- Where anything is outstanding, the holding's words are made new, its owner word is `MESH_STREAM_GIVEN` alone (it
  names no client, so none takes it), and the stream is not live for the rest of its link's pairing. No queue pair is
  moved to another state and nothing is awaited. A link none of whose streams is live is lost by the path every lost
  link takes (`broken`, the child's CHILD_BROKEN, port_lost, the child asked to end once its quiet window has passed):
  its exit releases its queue pairs, and the next child pairs from 0. The loss is booked as what it is,
  `LINK_STREAMS_ENDED` (ECONNABORTED) in the links table and the log, not a failed lane's EIO (rubric M10). On a link of two queue pairs (one stream) that
  follows at once, after the port's quiet window as every loss's. On a link of several streams, a dead stream's
  outstanding units stay posted, and the rows they name held, until they complete or that port's child ends, which may
  be the pairing's whole life: at most a lane's receive window and its SEND gate, each within a ring of
  `MESH_LANE_UNITS`. An end is decided at each end alone, so a stream can be live at one end and not at the other; an
  init takes only a stream free at both ends, so such a stream serves no communicator, and a link whose every stream
  is so is not lost by this path while each end keeps one live (open, for the operator: the peer is told nothing).
- Rows. A row is held only while it is in W(t): a unit outstanding on some lane names its page, a receive posted and
  not landed or a SEND posted and not done (mesh-dataflow.c `mesh_pinned`). That is the plan's window in flight, at
  most a lane's queue each way, and nothing else is held. A holder's halt (its give or its death) leaves L(P, t) empty,
  so it frees every row outside W(t) at once (design/extents-and-liveness.md, d): a lease of it is released at once,
  by the client's return (`mesh_lease_return`) or the bridge's release of a retired client's
  (`mesh_retired_release`). The reclaim reads each stream's owner word first, with acquire. Where it still names the
  holder and the holder gave or retired, the end is not yet published and the pass under way may post more of the
  holding, so the lease is held while the child serving it lives. Otherwise the word read is the end's store or a later
  one, which follows every tell of the holding, so the counts and rows read after it name every unit the holding
  posted. A lease with a row in W(t), or returned before its holding's end was published, is closed instead
  (`mesh_lease_close`), and released once its units complete, or once the port child that posted them is gone. The
  bridge releases closed leases at its turn (`mesh_retired_release` of every client), and a client releases its own at an allocation
  that finds no room (`mesh_retired_release` of its own). A lease's `closed` word holds its state (live, closed, released, free) in
  its low two bits and its row's generation above them. Each move is a compare-and-swap from the one state it leaves
  to a greater value, and the next allocation of the row starts the next generation. So the word takes each value once
  at most, and each generation is closed once, released once and freed once. A release swaps from the closed value
  read before its pin check. Two releasers that read one closure release it once, and a releaser overtaken by a free
  and a new closure fails. The release sets the lease's FREE bit after its swap. The reclaim frees a lease where its
  fetch-and took that bit and its swap moved the lease from released to free. A retirement closes only a live lease
  (`mesh_retire`), so a lease its client released before it died is freed once (`mesh_retired_release`, `mesh_lease_free`). A
  work request's buffer is the hardware's until
  it completes or until the queue pair holding it is gone with the process that owns it (verbs), so each stream records
  that child's pid and start (`child`, `child_start`, as the client slots identify processes) and its units count only
  while that process lives. A lease returned while its holder still holds the stream and its presented entries still name it
  is in L(P, t): returning it is the holder's error, as a use of a freed buffer is.
- The flush. The operator permitted it on 2026-10-22: "'The standard way to complete outstanding requests without the
  peer is to move the queue pair to the Error state. Every pending request is then flushed with IBV_WC_WR_FLUSH_ERR,
  and buffers are reused only after those flush completions are polled.' you can do this and nothing else with error
  states and if and only if it is exactly what is supported by the literature." It is not used: it would only end W(t)
  early, the plan bounds W(t), and the child's exit ends it (design/extents-and-liveness.md, "What follows for the
  mesh" 3; rubric M3).
- The cost: a link's controller that has been quiet is slow on its next frame (6.2 us against 1.9 on the Mac mini,
  metal-microbench docs/measurement.md, 2026-10-17). Without the keepalive that is paid at a holding's first SEND and
  after every quiet stretch inside a holding (a layer's 40-50 us between crossings); on a plan with lane 0 the
  keepalive had kept the controller warm inside a holding too.

**A process a port** (version 130, its streams new at every pairing from 131; the operator's choice of 10-20).
- The bridge is a parent and a child a port. The parent, its heartbeat's child, holds the region, the map and records, the
  clients' slots, device discovery, hellos and the control sockets. A port's child is `mesh-flow` spawned again by the
  parent, told the version of the contract between them (`--child VERSION DEVICE ...`, mesh-bridge.h MESH_CHILD_VERSION,
  declared, and changed where the contract changes and not by comparing builds: one source builds different programs on
  different machines; the operator, 2026-10-21: "commits which do not change an interface 'contract' in any way are not
  interface-version changes regardless of their hashes"), in a
  process group of its own, its parent's socket at descriptor 3. It opens the port's device, registers the region,
  makes and connects the link's queue pairs (its plan's, from the queue pairs both ends' devices hold, each end's told
  in its HELLO: [HETEROGENEOUS-LINKS.md](HETEROGENEOUS-LINKS.md)) and runs the port's progress threads. The parent asks it to open,
  make, connect, halt and end; it answers each, naming the pairing it was asked for, and tells failed lanes and absorbed
  faults. Queue pairs exist only in a child, and a child is the bridge
  ("the bridge and only the bridge may ever have access to that exact data structure", 10-20).
- A link's queue pairs are made once and connected once in a child's life. None is moved to another state,
  destroyed or forgotten; the kernel releases them at the child's exit. A link takes every queue pair its device holds
  for it (ten of a macOS 26 device's ten; two of the M5 Ultra's macOS 27 device's two by max_qp − 1), so a returning peer's lanes
  cannot exist beside the old ones, and the old child's exit is their one release (RESET panicked the M5 on 10-19;
  destroy is ruled out by its history, M3).
- A link lost once its child was asked for lanes (the control socket's end, a failed completion, no stream of the link
  live, the link going off, a termination, an absorbed fault, a protocol error, the child's own end): its
  streams cancelled (landed
  words ~0), its lanes served no more (the child halted, before the control socket closes), the port closed. Once the
  port's quiet window and any termination's have ended and no call of a port is out past MESH_PLATFORM_NS (mesh-link.c
  ports_stalled), the parent asks the child
  to end, reaps it and spawns the next, which opens the port and pairs with the peer whenever it answers. Nobody acts.
  Other ports' children and links, the clients and their leases are untouched.
- A child that ends unbidden loses its link the same way and is spawned again once the port's quiet window has
  passed; the k-th such end in a row within 60 s of its spawn, k from 2, holds the port's next spawn 60·2^(k−2) s, at
  most an hour (the operator, rubric M6: "restarted, not crashing again of course"). A child that makes no progress for
  MESH_PLATFORM_NS, or to which a message did not go whole, loses its link the same way, is asked to end once its
  windows have passed, and counts as an unbidden end; a child asked to end that has not ended is stopped and then
  killed, outside every window, its port with no job out and no port's job out past the deadline (section 16). A child whose parent is gone halts its
  lanes and exits MESH_PLATFORM_NS later, its own windows held; a port's child is spawned only while no other child of
  its device lives (one of the launch before, whose parent ended unbidden), so a device's queue pairs are one child's at
  a time and no make meets the device's ceiling. At the bridge's stop every child is halted, asked to end after the
  exit's hold, and awaited up to MESH_PLATFORM_NS.
- Streams across children. A port's streams' lists, rings and words are the parent's, leased once. Each stream names
  the child that posts on it (`child`, `child_start`), and its units count only while that process lives: a child's
  exit, whether this launch reaps it or it is an earlier launch's, releases every row its units named, by no port
  index and with no wait for a next pairing (the parent looks at the closed leases again as it reaps its own). A new
  child opens every stream new as its link connects: owner word 0, its counts 0 (`ready`, `sent`, `landed`, its
  lists', `posts`), itself its poster, and live. A holding of the lost pairing
  ended with it: its holder's give (a compare-and-swap from its own client) then matches nothing, and its communicator
  reads its holding gone (mesh-call.h `mesh_stream_held`: live, its owner word its client's, its identity the one it
  took) and starts no group on it (libnccl-mesh `comm_failure`, `ncclCommGetAsyncError`). No holder keeps a stream from
  the link's next pairing by letting go or by never letting go (design/rubrics.md M4, 10-20). Nothing of the lost
  pairing is sent on the new one: that pairing's groups ended at their next look at a landed word ~0, the connect comes
  `MESH_PLATFORM_NS` and a child's exit after the loss, and the two ends of a communicator take only an identity free
  at both (`wire`, a digest of this pairing's queue pair numbers and addresses).
- A LINK names the address its lanes were made on (a make reads the port's address again, as a peer's restart can move
  its GID table); the connect and each stream's identity take the peer's address from its LINK.
- Plug to link, at the hardware's pace. A port is opened when its interface's link comes on, when its RDMA peer
  interface arrives (IOKit's first match of an `AppleThunderboltRDMAPeerInterface` whose RID is its interface's: the
  en link can come on before the RDMA peer, and an open then finds the device's port not yet active), at a hello heard on
  it, and when its last child has ended; an open port with no peer says its hello at once, again whenever an IPv6
  address of its interface is added or changed (the kernel's `KEV_INET6_SUBCLASS` events: a hello that left before the
  address could carry it), and every MESH_PAUSE_NS otherwise. The parent's event loop sleeps until the next event or
  the next time its turn waits for (a quiet window's end, the exit window, a hold, a hello, a job's bound, the liveness
  turn), at most MESH_PAUSE_NS; every other change it takes wakes it where the change is made (a
  child of an earlier launch is awaited by its exit, EVFILT_PROC). Between a cable's loss and its next pairing stand the
  quiet window (MESH_PLATFORM_NS), the old child's exit (the kernel's release of its device's objects), the new child's
  open and the make and connect: the
  operator's deadline and the hardware's calls, and no timer of the bridge's.
- Calls in flight on a lost link fail as values (ncclRemoteError) and the next call takes a stream of the new pairing:
  "a bridge crashing on a computer two hops down should result in the bridge being restarted, not crashing again of
  course, and becoming available again as a resource for collectives with nothing but an error channel indicating that
  calls-in-flight failed and ought to be retried" (the operator, 10-20).
- XMAGIC marks a message as the mesh's, at the value version 131 sends, fixed: no version enters it, so bridges of any
  version pair (the operator, 2026-10-21: "rejection filter removal. categorical. all of them without sanction in
  rubric"). A message states its sender's MESH_VERSION in `version`, which no reader tests. A client reads a region
  only at the layout it was built for (MESH_LAYOUT): every field of another layout lies at other offsets, so a client
  writing it would write the bridge's words at random; that is a bound on memory, not a version check.
  A change that keeps every exchange of a version's meaning and adds information only in a field that version sends
  as 0 keeps the version: the lane budgets of [HETEROGENEOUS-LINKS.md](HETEROGENEOUS-LINKS.md) ride a HELLO's `count`, a
  peer's 0 there meaning a bridge before them, which makes every lane, and two ends that both hold ten pair as before.

## 15. The source

The bridge is one program, rdma/mesh-link.c over rdma/mesh-bridge.h: `main` runs the parent and `child_main` a port's
child (`mesh-flow --child VERSION DEVICE INDEX ONES`). libmesh is rdma/mesh-dataflow.c over rdma/mesh-call.h. The
region's shared words are rdma/mesh.h, and the bridge's own published pages rdma/mesh-observe.h. Their comments are
citations of this document; what each part does is stated here and in the sections above.

**The region** (mesh.h).
- `MESH_VERSION` is the contract between bridges: HELLO, RECORD, LINK and each lane's units. The operator,
  2026-10-21: "commits which do not change an interface 'contract' in any way are not interface-version changes
  regardless of their hashes". 129: a link's streams toward its peer node, held by owner words. 132: every unit a whole
  page, sent from and landing in the page its row names. 133: a stream's lanes carry its holders' entries' pages and
  nothing else. 134: a holding ends with nothing outstanding, or its stream is not live again in the pairing. 135: no
  FAREWELL; a bridge's stop reaches its peers as its control sockets' end (RECORD and LINK keep their kinds, 3 and 4).
- `MESH_LAYOUT` is the region's layout. 139 put each writer's words of a stream on a line of their own; 140 counted a
  lease's closures; 141 holds a lease's state and generation in `closed`; 142 deletes the header's `control` (the word
  a client's retirement rang to wake the bridge's watcher thread), `busy` and `idle_since` (read by the converger
  alone, deleted with it in 5aafa852) and `stopping` (never read), and names each child's device in its holder entry.
- `mesh_layout` places, each part rounded up to a page: the header; the row planes (`MESH_ROW_OWN`, `MESH_ROW_HOT`,
  `MESH_FREE`, a bit a row each); the arena's occupancy bitmap, a bit an arena page (an index space of its own, apart
  from the rows'); the page table (`struct mesh_page_entry` a row: the arena page it maps and that page's region
  offset); the leases' heads (`struct mesh_buffer` a row); the bridge's own bytes (`MESH_BRIDGE_BYTES`); the streams;
  the client slots; then the arena, aligned to a block.
- An owner word is `(generation << 38) | (slot << 32) | pid`, its generation 25 bits (mesh-dataflow.c
  `mesh_slot_take`). Bit 63 is clear in it, and `MESH_RETIRING` (a slot's) and `MESH_STREAM_GIVEN` (a stream's) set it.
- A lease's `pages`, on its first row, are its arena pages. `mesh_lease_move` swaps `closed` from one state to the
  next within a generation. `mesh_lease_close` reads the state, then the owner, then swaps from live to closed, so the
  swap succeeds only where the generation it read stands, owned by the owner it read (`mesh_rows_alloc` stores a
  generation's live state before its owner, after the previous generation's free stored 0).
- `struct mesh_pages` (a count of pages) and `struct mesh_row` (a page-table row) are types of their own, so the
  compiler keeps them apart from each other, from bytes and from a lane's units (8fc0d3d7). The region's words keep
  their integer storage and are typed where they are read and written (`mesh_entry_row`, `mesh_pages_load`).
- The map is `map_nodes` records at `map_off`, in arena pages the bridge leases, and leases again larger as the mesh
  grows. The bridge notifies `map` before and after each change (odd while it writes); a reader's copy is whole where
  `map` was even before it and is unchanged after (`mesh_map_whole`). A record carries no host name: the name a
  person reads is the name table's, by node number (etc/mesh-nodes.json, rdma/peers.py).
- `MESH_PAUSE_NS` (3 s) is the pause between a port's hellos and between the bridge's liveness turns.
  `MESH_PLATFORM_NS` (5 s) is the operator's deadline of section 5.

**The bridge's own pages** (mesh-observe.h). The port table (`struct mesh_link_info` a port: its peer while paired,
its phase, the code it last stopped with, its bandwidth and the pairings it made) is published for mesh-stat and the
node services; no client header includes it. Beside it, `struct mesh_bridge_pages` records each port's link page and
streams' lists, the duds' block and the map's pages (`MESH_ABSENT` where none), so the next bridge adopts them with
the region. It also records the bridge's start and each child's pid, device (an FNV hash of its name) and start, in
twice as many entries as ports (an
earlier launch's child that has not ended keeps its entry), so the heartbeat unlinks an expired region only once none
of them lives. Its fields are only appended. `mesh_say` (mesh-observe.h) is the log line of the bridge and the
heartbeat: the wall clock (UTC, to the millisecond) and the uptime, then the caller's prefix and the message, in one
write.

**Windows.** A port's `quiet_at` lies MESH_PLATFORM_NS past its last loss or past a peer interface's termination heard
(`quiet_all` moves every port's and the exit's `exit_at`), and moves only later (`quiet_until`), by any thread. Before
a port's window ends no job is dispatched on it; before every window ends (`windows_end`) the devices are not read
again and the bridge does not exit. A job of a pairing since lost (`pairing` moved by a loss or a halt, `port_halt`)
makes no further verbs call and is dropped as it ends (`link_lapsed`). A port has one job at a time, the parent's and
its child's alike (`job`: open, make or connect). A port whose job is still out
MESH_PLATFORM_NS after it began is concluded around it once (`ports_stalled`: open 0, `port_lost` with ETIMEDOUT, no
verbs call); while any job is so out, no port dispatches a job, no child is asked to end and the devices are not read
(`stalled`), and the port stays `stuck` until its job returns, then opens again. The event loop's wait (`turn_due`)
ends at the earliest time a turn waits for (a liveness turn, a health sample, a window's end,
a spawn hold, a hello, a job's bound), at most MESH_PAUSE_NS on; every other change wakes it where it
is made (`wake`, a socket, a signal, an exit). A fixed 250 ms tick had taken each such time
up to a tick late.

**The data path** (mesh-link.c, a child's progress threads). A child runs two threads, `way` 0 for its SENDs
(mesh.rdma.send) and `way` 1 for its receives (mesh.rdma.receive), the same index as the region's `posts[way]`. A pass
(`port_pass`) reads the lanes served once, takes each stream's holding (`stream_turn_one` at the send thread,
`stream_receive_turn` at the receive thread), posts its units (`stream_post`), then polls that way's completion queue.
A thread with no lane served waits on `served`, at most MESH_PAUSE_NS a wait, and reads `closing` between waits.
- The two threads hand a holding over by its number: the send thread begins holding n (`begun`) and ends it (`ended`)
  at the first pass that reads its give or its client's retirement, and frees or ends the stream once the receive
  thread has stored `stopped` = n after its last post of it. The receive thread serves holding `rnumber` of owner
  `rheld` (the send thread's `held` as it began), and passes over a holding that ended before it began.
- A walk (`struct walk`) takes a list's presented entries into a ring of its own (`walk_take`: each entry copied once
  and typed as `struct taken`, `whole` 0 where its rows lie outside the page table or a send entry's word lies outside
  the region), finds the entry holding a page (`walk_page`), stores its count (`walk_count`: `sent` or `landed`, with
  release, first in the pass) and passes the entries whose pages have all completed.
- Page p of a stream is unit p div 2 of its lane p mod 2 (`page_lane`). Its pages posted are its two lanes' units
  together (`stream_pages`), and its pages completed in order run to the first page either lane has not completed:
  lane 0's 2a or lane 1's 2b + 1 (`stream_through`).
- `page_span` is the one place a page is bytes: the scatter/gather entry of a page at an arena offset, its length the
  rest of that page's block from the offset, its key its extent's. `row_at` reads a row's page from the page table, 0
  where the row holds no page of the arena (a row with no page reads address 0), so no entry names memory outside the
  registered arena.
- After the completions, `walk_complete` stores each served holding's count first, then reads the clock once
  (`mach_absolute_time`) where a SEND was posted or an entry landed whole, then tells each lane's units completed
  (`posts`), then fills the time rings: a send entry at its first page's post, a landing entry at its last page's
  landing (`stream_time`: the slot's first word 0, then the ticks, then 1 + e, each a release store).
- A lane's receives run at most `window` units ahead of its landings: the fewer frames of the lane's share of the
  completion queues and of its receive queue, over a unit's frames, at most MESH_LANE_UNITS; its SENDs at most `sends`
  units ahead of their completions: the fewer frames of its share and of its send queue, over a unit's frames, at most
  MESH_LANE_UNITS (`link_connect`).

**Completions.** The device completes every SEND, signalled or not, so a SEND's completion is its queue's retirement.
The census of e2a29067 (requests counted against completions) had no reader once its log was removed, and is deleted.
A completion's status and opcode are read as one 8-byte word (`struct ibv_wc`'s status at offset 8 and opcode at 12:
mesh-bridge.h's static assertion). A completion on a lane no longer served is dropped; a failed one, or a failed poll,
ends serving on every lane of the link, and a failed post on its lane (`lanes_lost`), which the event loop then loses
(`broken`).

**The SEND gate.** `ibv_post_send` accepts posts past the queue's frames and corrupts later, so the send thread posts a
lane's SEND only while that lane's SENDs posted less those completed (`sent` − `done`) are fewer than its `sends`. The
gate of 5d6f8c61 counted frames, for transfers of any size; since version 132 every unit is one page, so its frames were
these units times a constant, and its own counts and `wr_id` encoding were deleted (2026-10-22).

**The children.** A child is spawned with its parent's socket at descriptor 3, /dev/null as its input and output and
its parent's stderr, in a process group of its own (`child_spawn`). The parent's end of the socket does not block: a
message that does not go whole marks the child deaf (`child_say`), and nothing more is sent to it (the child would read
the next message's bytes as the rest of the cut one); a job whose message it was ends EPIPE at once. Each child's holder entry names its device, so a
bridge finds a child of an earlier launch that still holds a device in the region it adopted, and spawns its own after
that one's exit (EVFILT_PROC); every child of a region is in its entries, since the heartbeat opens a bridge only once
no holder of an expired region lives. `struct child_message` carries, from the parent: open; make (the link's page and
plan); connect (the peer's lanes, address and node); halt; end. From the child: each job's end, its kind the job's
plus CHILD_OPENED − CHILD_OPEN, naming the pairing it was asked for (opened: its address, bandwidth and the queue pairs
its device holds for a link; made: its address, each lane's queue pair number, the first sequence number and the
lanes' share; connected); the lanes found failed (`CHILD_BROKEN`, its `result` LINK_STREAMS_ENDED where no stream is
live); a fault absorbed (`CHILD_FAULTED`). After a termination it hears itself, the child answers make and connect
with ECANCELED until the parent's halt (the parent hears the same termination from IOKit itself), and after a fault
for the rest of its life. Each turn of its main loop at which each progress thread serving lanes has passed since the
last such turn, it writes the clock to the region's `progress[port]` (section 16).
- `MESH_CHILD_VERSION` 7 has no `refused` in a child message (the registration-refusal backoff deleted). Version 6
  writes the child's progress, which its parent reads, and ends the child after the provider's
  fault absorbed in it, as a child whose parent is gone ends. Version 5 told an absorbed fault (`CHILD_FAULTED`, after `CHILD_BROKEN`) and has no
  `CHILD_TERMINATED`; its open names its pairing and its connect no bandwidth (the child keeps its own from its open).
  Version 4 (ad55cbf4) had no `CHILD_FAULTED`. Version 3 had no `CHILD_UNGUARDED`. Version 118's guard (542d0bc7) held a port's progress threads
  around every verbs change, for stripe flows whose queue pairs were made and destroyed while other queue pairs of the
  same device were posted on. Since version 130 a child makes and connects its queue pairs before it starts its
  progress threads, and no later open, make or connect reaches a child whose threads run: the parent marks the lanes
  asked for (`lanes`) before the make, and a link lost after that ends its child (`recycle`). No guard was raised with
  a thread running, so the guard, its waits and epochs and `CHILD_UNGUARDED` were deleted (2026-10-22).
- A make reads the port's address again first: a peer's restart can move the port's GID table, and on 2026-10-19,
  after the Mini's power loss, LINKs naming the address of the port's opening failed every lane's connection at the
  peer with EIO. It then makes a UC queue pair in INIT for each lane of the plan (d6dceac's verbs_up attributes), and
  the connect moves each to RTR, then RTS.
- A stream's identity (`stream_wire`) is FNV-1a over its first lane's two ends (each port's GID and queue pair
  number), the lesser end first, 1 where the hash is 0. A stream's bandwidth is the link's over MESH_LINK_STREAMS,
  however many of the link's streams are live, so the map's figure for a link is alike for links of every plan.
- A child whose parent is gone halts its lanes and exits once MESH_PLATFORM_NS and its windows have passed
  (`child_leave`), at most twice MESH_PLATFORM_NS on.

**The port's device** (`device_up`, `device_gid`). A port's device is opened by its name from the device list, its port
read, its protection domain made and the whole region registered in its extents (section 1). A registration that
fails ends the open with its errno, as a value: the port is not open, the extents taken stay the device's, and the
port opens again at the events that open any port (its link's coming on, a hello heard on it, its child's end), when
the rest are registered. The registration-refusal backoff of 128d0e84 (EFBIG past `max_mr`, the k-th refusal in a row
held 60·2^(k−1) s) is deleted: the extents fit `max_mr` by construction, and a hold that retries a deterministic
outcome serves none of AGENTS.md "Recovery"'s three kinds of failure (the operator, 2026-10-22: "all of the relatd
abstractions and wrappers must be deleted"). The queue pairs usable for a link are `max_qp` − 1. The port's GID is chosen
from its whole table (up to 256 entries): the entry equal to an IPv6 link-local address its interface holds
(getifaddrs, with the scope BSD embeds in bytes 2-3 cleared), else the table's link-local entry, else its first nonzero
one, the open logging that no address matched. A Thunderbolt port's table does not always begin at index 0: the Mini's
en4 after its 2026-10-16 restart held its only entry at 1, and index 0's empty address failed every queue pair's
connection with EIO. The M5 Ultra Studio's macOS 27 rdma_en3 (2026-10-21) held [0] 23ff:9f:174a:401d:8d37:9dff:16ac:4594,
[1] its interface's fe80:: link-local address and [2] its fd6d ULA, and the Mini (macOS 26) failed every lane's
connection toward [0] with EIO.

**The control socket** (`port_say`, `control_heard`). A message is a header and its `bytes` (at most 64 MiB), queued
behind what the socket has not taken, with no bound: a peer that reads nothing holds a record a change or a health
sample of each node until the queue cannot grow. A failed send, or a queue that cannot grow, loses the link. An open
port with no peer multicasts HELLO to ff02::1 on its interface every MESH_PAUSE_NS, and again as an IPv6 address of
its interface is added or changed (`addresses_heard`, matching the event's interface family and unit to the port's,
read once from its name; another Ethernet-family interface of the same unit counts as the port's). The lower node dials
the higher's listener at the address it heard from, and the higher answers its hello. A HELLO on the control socket
states 1 + the queue pairs its device holds for a link. A peer whose blocks are another size loses the link (its SENDs
would cover other frames than this end's: TN3205), and so does a LINK naming other lanes than this end's plan.

**The map** (`map_put`, `record_heard`). This node's record is made again when its port table changes (its sequence one
more) and told on every greeted port (`records_say`), and a port newly paired is told every record of the map. A record that changes only its health and
sequence is written under the `health` serial, not the map's, so the map serial moves only where the mesh changes. A
record put in is kept only where a paired path from this node reaches its node (a breadth-first walk over
`mesh_joined`); one the walk drops is logged unreachable. `known` is RFC 2328 s13's database: each node's newest record
heard (its boot and sequence), held apart from the map, so a record no newer than one heard is not flooded again; a
record the map dropped is put back when heard again at that sequence (a new pairing's exchange of the map) or a newer
one (its node's next change or health sample, at most MESH_HEALTH_NS on). A full table gives its longest-unheard entry's
place. A record heard is copied up to this build's size, the rest 0
(`struct mesh_record` is only appended to). A record's reach is every cable whose link is up toward a node with a
stream in its plan, their bandwidths summed, so two nodes several cables join are one link of the map. Its health is
the node's boot time and its available memory (`kern.memorystatus_level` percent of `hw.memsize`), sampled at most
MESH_HEALTH_NS (10 s) apart.

**Terminations and faults.** IOKit tells a Thunderbolt RDMA peer interface's termination in its first phase
(kIOTerminatedNotification), on a queue of its own: every window is held MESH_PLATFORM_NS from then before any property
is read, and the port whose interface carries the same "RID" (bin/mesh-fabric-init.sh) is halted. The parent also hears
arrivals (kIOFirstMatchNotification) and opens a port whose device's far end has come, which the en interface's link
can precede. The provider's fault is absorbed in the port's child (`fault_heard`; 2f4c511f, "the bridge does not
exit"; b53804bf; section 16): the child halts its lanes, answers ECANCELED a job the fault ended, tells its parent once
(`CHILD_FAULTED`), which loses the port as at its link's going off (open 0, ENETDOWN) and does nothing else, and ends
itself as a child whose parent is gone does (`child_leave`): MESH_PLATFORM_NS and its windows after, at most twice
MESH_PLATFORM_NS. Every thread of it ends with it. Its reason is rubrics.md "Failure across interfaces" 4 ("inducing a
system halt or kernel halt causes the entire node to fail. that is not permittable for obvious reasons"): a child that
ended at the fault would exit, and its kernel release its device's objects, inside the termination's window (section
5). ad55cbf4 deleted the absorption, and it was restored the same day.

**The region** (`region_open`, `region_adopt`). A region is adopted where its magic, layout version and geometry (page
size, block, rows, stream and client counts, every offset and its length) are this bridge's. Adopting it sets every
stream not live, its peer and identity 0 and its `sent` and `landed` ~0; clears the port table; finishes a retirement a
dead process left half done (`mesh_retire` of an owner with `MESH_RETIRING` set, the walk any party can repeat to the
same end); releases closed leases; makes an odd map or health serial even; and takes the map's pages again. A shared
memory object without the magic (a bridge ended while making it) is unlinked and made anew. The bridge's node number is
a 31-bit FNV hash of the host's UUID (drawn at random where none is read), 1 where the hash is 0.

**The library** (mesh-dataflow.c, mesh-call.h).
- `mesh_open` maps the region read-write (a client's) or read-only (a reader's), EINVAL where it is no region of this
  layout. A slot is taken vacant by one compare-and-swap and stamped with its process's start; `mesh_attach` reads the
  expiry again after the swap, so a slot the heartbeat's read of the holders missed reads its region expired and lets
  it go.
- `mesh_claim` claims a range found clear word by word, giving the words back where another client claimed first or a
  row is still hot (released and not yet freed). The allocation cursor resumes where the last allocation ended (O(P)
  over N allocations of P pages), and an allocation that finds no room releases its client's own closed leases and
  scans again.
- A lease is freed by its one freer: its rows' owner words 0, then its pages' arena bits, then its rows' hot bits, so
  no row or page of it is claimed before it is whole. `mesh_retired_release` releases the closed leases of one client,
  or of every client at the bridge's turn, where `mesh_pinned` finds no unit naming them, then frees each lease whose
  FREE bit its fetch-and took and whose `closed` it swaps from released to free. A retirement (`mesh_retire`, the
  client's detach or the bridge's for a client gone) ends with that release of its client's own.
- The map's readers (`mesh_map`, `mesh_nodes_health`, `mesh_map_wait`) wait on an odd serial at most MESH_PAUSE_NS a
  look, and return ESRCH where the bridge went meanwhile. `mesh_joined` is the one join of two records, the bridge's
  map walk's and the readers' alike: each reaches the other, at the lesser bandwidth either reports.
- `mesh_stream_take` returns EBUSY where another client holds the stream or its last holding is still ending, and
  ENOLINK where no live stream has the identity. `mesh_stream_present` returns EAGAIN where MESH_STREAM_ENTRIES entries
  are presented and not taken, and EINVAL where an entry's rows leave the page table.

**The heartbeat** (rdma/mesh-heartbeat.c). Its signal set (`heard_set`) is built in one place for its mask, its spawn's
defaults and its re-raise. Before it runs its image again it blocks its signals and raises again each
one its kqueue heard since its last read, so the signal is pending across the exec. While an expired region waits for
its holders, the slots are read every second for a client that lets go and lives on. A bridge asked to end that has
not ended 3·MESH_PLATFORM_NS later is left, and the heartbeat ends. While a bridge is open its watchdog (`watchdog`,
TIMER_WATCH each WATCH_MS) reads the bridge's `turned` and `hung_from`, adds to the statistic and ends a bridge whose
loop has stopped (section 16).

**mesh-stat** prints one JSON line: `up` (else the errno), the node, its ports by index (peer, phase, bandwidth), the
map (nodes by bridge number; links as [node, node, bandwidth or null]), the region and its unleased bytes, and each
node's record (region, memory, model, boot, available memory), and the hang statistic (`hang`: `active` and `hung` in
nanoseconds, `hangs`, and `share`, hung over active: section 16), printed beside `up` false too (0 where no region stands).

**The disk floor** (rdma/mesh-disk.h). MESH_DISK_FLOOR_GIB is the free space a writer of the mesh stack leaves on its
filesystem; rdma/mesh_disk.py and the services' scripts read the number from the header. The log cap beside it
(MESH_LOG_CAP_MIB) lost its last reader in 4d4595cc and was deleted.

## 16. Hangs, and the watchdog

The operator, 2026-10-22: "lets focus on making sure peers: 1: don't hang 2: are heartbeated and or watchdogged along a
distinct hardware process so they unhang if they somehow hang anyways 3: are hanging less than 0.1% of the time a
bridge is active". The deadline is section 5's ("you cannot be using a low level kernel like a dma or networking
send/rcv channel for more than 5 seconds without something happening/concluding"), and the stake THREAT-MODEL.md's
("the most important class of compromise is the false negative: the computer should be on and receiving work and doing
work, but isn't, for any reason").

A hang is a party with work outstanding whose progress has not moved for MESH_PLATFORM_NS. The parties are the
bridge's processes: the parent; each port's child, with its main loop, its job thread (the port's serial queue) and its
two progress threads; and the heartbeat. A hang is remedied by acting on the hung party itself: its own supervisor ends
it (the heartbeat the parent, the parent each child), the next one starts, and its peers pair again unaided (section
14). Nothing fails a client's program: calls in flight on a link the remedy loses fail as values, as at any link loss.

### Every wait and loop

The parent (mesh-link.c `main`):

| wait or loop | waits on | ended by | why it ends |
|---|---|---|---|
| the event loop's `kevent64` | signals, sockets, children's exits and messages, `wake` | the event, or `turn_due` | the timeout is at most MESH_PAUSE_NS |
| a turn's work (`job_finish`, `port_open`, `port_hello`, `link_turn`, `child_turn`, `clients_turn`, `map_put`) | nothing | its end | loops over the ports, slots, rows, streams and lane rings, each a fixed table |
| `quiet_until` | other writers of the word | its swap, or a value at least `until` | each failed swap read a larger value; the word only grows |
| sends on a control socket (`port_flush`) | the socket's room | EAGAIN, then EVFILT_WRITE | non-blocking |
| sends to a child (`child_say`) | nothing | at once | non-blocking (since this section); a send that does not go whole marks the child deaf (below) |
| `recv`, `recvfrom`, `accept`, `connect` | data | at once | non-blocking sockets |
| the device list's read (the `mesh.ports` queue) | the kernel | its return | not ours to bound: read only outside every window and while no job is out past the deadline; past the deadline it counts as hung |
| the IOKit notifications (`peer_terminated`, `peer_arrived`, `peer_port`) | the kernel | its return | not ours to bound |
| `waitpid` in `child_reaped` | the child's exit | the exit | called on NOTE_EXIT, or where that registration failed |
| the stop's hold (`poll` on the link events) | the windows' end | its bound | MESH_PLATFORM_NS in all, each wait at most MESH_PAUSE_NS |
| the stop's wait for its children (`kevent64`) | their NOTE_EXITs | its bound | MESH_PLATFORM_NS in all, each wait at most MESH_PAUSE_NS; a child it had stopped (below) is sent SIGKILL first |
| the log (`say`) | the log file | the write | launchd's StandardErrorPath, a regular file |

A port's child (`child_main`):

| wait or loop | waits on | ended by | why it ends |
|---|---|---|---|
| the main loop's `kevent64` | its parent's socket, `wake` | the event or the tick | MESH_PAUSE_NS |
| sends to the parent (`message_send`, blocking) | room in the parent's socket | the send | the child sends one reply a job (a port has one job out), one CHILD_BROKEN and one CHILD_FAULTED a pairing: at most three of 192 bytes unread in 8,192; a socket that full is a parent that does not turn, which its heartbeat ends, and the send then fails (EPIPE) |
| a job (`device_up`, `link_make`, `link_connect`, `getifaddrs`) | the kernel | its return | not ours to bound: the parent concludes the port around it at MESH_PLATFORM_NS (`ports_stalled`), and it counts as hung |
| a progress thread's pass (`port_pass`) | nothing | its end | posts and polls return (section 6); `stream_post` ends at the lane's window or the last entry presented |
| a progress thread with no lane served | `served` | a change, or the bound | MESH_PAUSE_NS a wait |
| `child_leave` | the windows | its bound | at most twice MESH_PLATFORM_NS, each wait at most MESH_PAUSE_NS, its progress written between |

The heartbeat (mesh-heartbeat.c):

| wait or loop | waits on | ended by | why it ends |
|---|---|---|---|
| `kevent64` | signals, NOTE_EXITs, its timers | the event | while a bridge is open its watchdog ticks each WATCH_MS; while a region waits for its holders, each REPLACE_TICK_MS |
| `waitpid` in `bridge_ended` | the bridge's exit | the exit | on NOTE_EXIT, or a failed registration of a bridge just spawned (`open_bridge`, the block-quoted spawn, unchanged), which is ESRCH where the bridge has exited |
| `run_again`'s drain | pending events | at once | a zero timeout |
| a bridge it asked to close | its NOTE_EXIT | its exit, or the watchdog | TIMER_END ends an end's wait after 3·MESH_PLATFORM_NS; the watchdog ends every ask's bridge that stops turning |

### The hang modes on record

| record | what hung | the form that makes it unreachable, or the watchdog that ends it |
|---|---|---|
| 2026-09-05 | processes in the release of their verbs objects at exit, during a device's fault (STAT U) | No process of ours exits inside a termination's window: a child is asked to end, or ended by the watchdog, only once its port's quiet window and every termination's have passed (`quiet_at`, `exit_at`) and while no job is out past the deadline (`stalled`). The exposure left is a termination that begins after that check (section 9). Nothing in user space ends a thread in U (below). |
| 2026-09-05, 10-18 | `deallocPD` waiting on a termination that waits on its gate | No `ibv_dealloc_pd`, `ibv_dereg_mr`, `ibv_destroy_qp`, `ibv_destroy_cq` or `ibv_close_device` exists in the bridge (section 5): unreachable. |
| 2026-10-02 | a GPU kernel spinning on a word another queue wrote | The collective's: every GPU wait keeps its poll bound. Not the bridge's. |
| 2026-10-17 | `pair_flush` waiting for drain markers | No flush and no drain exist: unreachable. |
| to 2026-10-19 | host and GPU waits on landed words, waits on peers in setup | The collective's: a group's waits re-read the revoke and the holding, the bootstrap's phases end within the deadline. The bridge waits on no peer. |
| 2026-10-16, 10-17 | the Mini's kernel data aborts at 0x98 seconds after its bridge stopped (a panic, not a hang) | Each stop then destroyed queue pairs while posts could be in flight. No queue pair is destroyed; a child's are released only by its exit, after it has halted its lanes (`served` 0, `closing`) and waited MESH_PLATFORM_NS and its windows (`child_leave`), so no post is in flight at the release. A child the watchdog ends cannot halt itself: it is stopped first (SIGSTOP stops every thread of it, so none posts), and killed MESH_PLATFORM_NS later, its device quiet as after a halt. The trigger is not established and the record stays open. A panic is the kernel's: no watchdog of ours runs through it; the node restarts, launchd starts the heartbeat, and peers pair again (M6). |
| this section's inventory | the parent blocked sending to a child that does not read (a blocking socketpair) | Non-blocking: a send that does not go whole marks the child deaf; its job, if that message was it, ends EPIPE at once (it never reached the child), nothing more is sent to it, and it is ended as a hung child. |
| this section's inventory | a child asked to end that does not end (its main loop not reading CHILD_END, or its exit held): its port waits for it forever (`recycle`) | The watchdog: SIGSTOP MESH_PLATFORM_NS and MESH_PAUSE_NS after the ask (`child_leave` ends an asked child within MESH_PLATFORM_NS), then SIGKILL MESH_PLATFORM_NS later, each outside every window, its port with no job out and no port's job out past the deadline (`child_may_end`). |
| this section's inventory | an earlier launch's child that does not end: its port's next child is never spawned (`elsewhere`) | The watchdog: the same, from twice MESH_PLATFORM_NS after it was found (its own `child_leave` ends within that), while it is still that process (its pid and start, `mesh_holds`). |
| this section's inventory | the heartbeat waiting forever for a bridge it asked to restart or release | The watchdog: the stop's waits turn the progress word, and a stop that stops turning is ended. |
| review of b53804bf (finding 6) | the provider's fault, absorbed in the port's child, recurring in it until its parent ended it, and holding a job of it out | Absorbed in the port's child, and ended within the child's own life: a job it ended is answered ECANCELED, and the child ends itself as a child whose parent is gone does, at most twice MESH_PLATFORM_NS on; the parent loses the port as at a link's going off and does nothing else. |
| any | a parent whose event loop stops turning; a child with no job out whose main loop stops, or whose progress thread stops passing while its link serves | The watchdog (below). |

### The watchdog

The words (rdma/mesh-observe.h `struct mesh_bridge_pages`, appended; each written by one process, read by its
supervisor):
- `turned`, the parent's: its clock at each turn of its event loop and at each pass of its stop's waits. The clock,
  the bridge's and the heartbeat's alike, is CLOCK_UPTIME_RAW (mach_absolute_time's), which stops while the system
  sleeps. With it, after it, `turned_by`: the writer's start (`bridge_start`), so that a stamp is read only as the
  stamp of the bridge that wrote it; the region outlives its bridges.
- `progress[port]`, each child's: the clock at a turn of its main loop at which each of its progress threads that
  serves lanes has passed since the last such turn. A progress thread counts its passes on its own line (`struct
  progress`'s `passes`), one store a pass and no clock. A thread with no lane served waits on `served` and holds no unit
  (a lost link's units stay the hardware's until the child's exit), so the passes counted are the posting threads'
  while units can be outstanding.
- `hung_from`, the parent's: the earliest time a party it supervises passed the deadline without progress, 0 where
  none (and 0 while it stops): a child's last progress, a job's start or the device list's read, each plus
  MESH_PLATFORM_NS; a child asked to end, from MESH_PLATFORM_NS and MESH_PAUSE_NS after the ask; an earlier launch's
  child, from twice MESH_PLATFORM_NS after its discovery.

What reads them, and what each reader does:
- **The heartbeat** (a distinct process launchd keeps; it holds no device) reads `turned` each WATCH_MS while a bridge
  is open, where `turned_by` names that bridge's own start (`mesh_started` of its pid) and, for a bridge it spawned,
  `turned` is later than the open. A stamp another bridge left in the region (a bridge found serving before its first
  turn, a build without these words adopting a region a later one stamped, a pid given again) is never acted on. The
  parent's loop always has a turn due at most MESH_PAUSE_NS on (its liveness turn), with or without work, so a quiet
  bridge turns and is never ended; where `turned` is MESH_PLATFORM_NS old, a turn due has not come, and the heartbeat
  logs it and sends SIGKILL. The bridge's stop signals are kqueue events and ignored as signals, so a loop that does not
  turn never reads SIGHUP or SIGTERM, and SIGKILL is the signal that ends it from every state.
  It holds no device, so its exit releases no device object, wherever a termination stands. Its children see its socket
  end, halt their lanes and leave after their own windows (`child_leave`); the next bridge adopts the region. The
  heartbeat asks nothing of its own: where it had asked nothing, the end is an unasked end, and the bridge opens again
  REOPEN_S after it, at most once a minute, as after a crash; where it had asked a restart, a release or an end, that
  ask is carried out. The heartbeat's reopen, backoff, spawn settings, signal branches, end timer and log format are as
  they were (the operator's block quote, 2026-10-21). The heartbeat acts on the parent's own loop alone: a port's stall
  is the parent's to end, by ending that port's child, which it spawns again; the whole bridge is never ended for it.
- **The parent** reads each child's `progress[port]` at each turn (`child_turn`). A child with no progress for
  MESH_PLATFORM_NS, or deaf, is hung: its port is lost (ETIMEDOUT; streams cancelled, calls in flight fail as values),
  a job of it that is out stays out, under `ports_stalled`, until it is answered, and the child is asked to end as at
  any lost link, once its windows have passed, its port has no job out and no port's job is out past the deadline
  (`child_may_end`). A child asked to end that has not ended MESH_PLATFORM_NS and MESH_PAUSE_NS later (its own leave
  ends it within MESH_PLATFORM_NS), and an earlier launch's child twice MESH_PLATFORM_NS after it was found, is sent
  SIGSTOP under the same gate, and SIGKILL MESH_PLATFORM_NS after that. A hung child's end counts toward its port's spawn backoff as an unbidden end does
  (section 14); its loss stays booked ETIMEDOUT.

### A thread in the kernel

A thread in a kernel call that waits on something that never happens (STAT U) takes no signal, SIGKILL included, and
its process's exit completes only when the call returns (section 5). For it the watchdog:
- counts the time, through `hung_from` (a job or the device list's read past the deadline) or `turned` (a parent's
  loop);
- ends a parent so held all the same: the parent holds no device, so its exit, whenever the call returns, releases
  nothing of a device, and the heartbeat opens no bridge beside it before its NOTE_EXIT;
- ends no child with a job out, nor any child at all while a job is out past the deadline (`stalled`): its exit would
  release device objects whenever the call returned, perhaps inside a termination. While such a call lasts no port
  opens, makes or connects and no child is ended (section 15, "Windows"), and the time counts as hung;
- reboots nothing, panics nothing and makes no teardown call. "transactions can fail, nodes that are healthy can never
  fail ... inducing a system halt or kernel halt causes the entire node to fail. that is not
  permittable for obvious reasons" (rubrics.md "Failure across interfaces" 4, 10-20). Its remedy is upstream: no call
  of ours that can wait on a termination (section 5), so the U state is reachable only through the kernel's own defect.
  The node stays up and reachable, the ports' children keep serving the links they have, and the parent serves its
  control sockets and the map.

### Peers

A peer can notice that this node's bridge is not answering: its control socket unread, its records not renewed, its
SENDs held in credit. Nothing is gained by having it act:
- only the node can end its own bridge; the peer would reach it over the slow channel, which "never decides a running
  RDMA channel's state" (rubrics.md "Two channels") and takes 10⁴ to 10⁸ ns where the node's heartbeat acts within
  MESH_PLATFORM_NS and a second;
- a parent that stops leaves its children serving: a peer that concluded the link would fail calls the data path is
  still completing;
- the bridge has no API to act on a suspicion (rubrics.md M1, "What the bridge cannot do": "declare a quiet peer lost"),
  and a timer that declares a quiet peer lost and fails the program ranks below no such detector (M4, 10-18; C5's lower
  pole).

In Chandra and Toueg's terms (1996), a failure detector is a module at each process whose suspicions only that process's
own algorithm consumes, by ceasing to wait for the suspected one; it never acts on the suspected process, and it may
suspect a correct one. Here no algorithm consumes a suspicion: the collective waits on landed data and the platform
deadline alone (C5), and the hung party's own supervisor is the one that acts. A peer reads a node's hangs in that
node's statistic and records, for people.

### Where a driver's or a link's event reaches

A device's or a link's event (a provider fault, a link's going off, a termination, failed lanes, a child that ends)
reaches, past the port's child and the parent's record of that port:
- the calls with positions in flight on that link: its streams are cancelled (landed and sent ~0, not live), and those
  calls fail as values (section 14);
- this node's record and the map: the link leaves the record, and the next calls are planned over the cables left
  (section 15, "The map"); that is data, and fails nothing;
- the peer's end of the link, by its control socket's end: its in-flight positions on that link fail as values;
- the rows its outstanding units name (W(t)), held until they complete or the child exits (section 14).
And, inside the bridge, a job past the deadline holds every port's jobs, device reads and child ends (`stalled`, section
15 "Windows") while it lasts; it fails no call and no other namespace. Nothing else is reached.

### The statistic

Three counters in `struct mesh_bridge_pages`, the heartbeat their one writer, each WATCH_MS while a bridge it watches
is open: `active`, the nanoseconds a bridge was open; `hung`, the nanoseconds of those past the deadline without
progress, from the earliest of `turned` plus MESH_PLATFORM_NS and `hung_from`; `hangs`, the times that condition began.
They live with the region, so they count across the bridge's restarts and begin again at a release. mesh-stat prints
them and `hung` over `active`; the operator's 0.1% is that ratio below 0.001, read once nodes run.

### Ranked

Each design point, by design/rubrics.md, the operator's words behind it, and the literature:
- **Who acts on a hang** (M4, M1, "Two channels"). A peer's or a client's detector declaring a bridge hung, or asking it
  to restart over the slow channel ≺ the hung party's own supervisor ending it, its peers pairing again unaided. "a
  bridge crashing on a computer two hops down should result in the bridge being restarted, not crashing again of course,
  and becoming available again as a resource for collectives with nothing but an error channel indicating that
  calls-in-flight failed and ought to be retried" (10-20). Armstrong's supervision trees (*Making reliable distributed
  systems in the presence of software errors*, KTH 2003; Erlang/OTP's supervisor): a supervisor starts, watches and
  restarts its own children, and a worker's failure is its supervisor's to handle, not its siblings'.
- **A distinct process** (the operator's item 2). A party watching itself, a thread of the process that hangs ≺ a
  watchdog in another process the hung one cannot stop: the heartbeat for the parent, the parent for each child.
  Mahmood and McCluskey, "Concurrent error detection using watchdog processors: a survey" (IEEE Trans. Computers 37(2),
  1988): a watchdog processor checks the main processor from outside it, from information the checked one emits as it
  runs.
- **How progress is read** (M2, M8). A probe the watched party answers (a message about messages) ≺ words it writes as
  it passes, read where they lie: "the only exchange that is not data is the pairing of a new cable" (M2). A clock read
  each pass of a progress thread ≺ a counter on the thread's own line, the child's main loop reading the clock it
  already reads: "if there is 10ns of reducible overhead anywhere in our code we reduce it to 0ns" (M8, 10-20).
- **Ending a hung parent** (M3). SIGHUP and a wait with no bound (the heartbeat before this section) ≺ SIGKILL at the
  deadline: "imagine a number for every isngle state the bridge cna be in. for all numbers there must be a signal whic
  hcan be sent to the bridge which causes the bridge to shut down as an application and release all resources" (10-18);
  "semaphores allow all waits in all programs to termiante in finite time" (10-20). Erlang/OTP's shutdown asks a child
  to end and kills it after its shutdown time; a loop that does not turn reads no ask, so the ask is skipped. Candea and
  Fox, "Crash-Only Software" (HotOS IX, 2003): a component whose stop is a crash and whose start is its recovery; the
  bridge's region is adopted across any end (section 2), so a kill loses nothing a stop keeps.
- **How soon it opens again** (the operator's block quote of 2026-10-21; rubrics.md "Failure across interfaces" 5 and
  6). A watchdog with a restart path of its own, faster than an unasked end's ≺ its end taken as an unasked end, the
  heartbeat's reopen unchanged: "launchd heartbeat reopen code is required by the user by block quote and will never be
  overridden implicitly as a side effect of any other message or statement"; "THE HEARTBEAT SERVICE THAT ENSURES UPTIME
  OF THE MESH SERVICE WHILE STILL IMPLEMENTING A BACKOFF SCHEDULE IF IT CAUSES A BOOTLOOP. there is only ONE program in
  this ontology which is capable of bootlooping or crashing other systems! it is the bridge!" (10-21); "a restart policy
  of one second ≺ one bridge start a minute at most" (10-20). OTP bounds restarts too (MaxR in MaxT) but then ends the
  supervisor itself, which here would leave the node with no bridge (THREAT-MODEL.md's false negative); here the excess
  waits instead.
- **When it acts** (the operator, 10-21, of the deleted keeper: "i do not think anything i said implied that i wanted an
  arbitrary unspecified service trying to crash services once a minute"). A watchdog ending a bridge on a schedule, or
  for quiet ≺ one ending it only where a turn due has not come for the platform deadline: a bridge with nothing to do
  turns each MESH_PAUSE_NS and is never ended.
- **The narrowest act** (M4). Ending the whole bridge for one port's stall ≺ ending that port's child, which the parent
  spawns again; the heartbeat acts only on the parent's own loop, which no child can end.
- **Ending a hung child** ("Failure across interfaces" 4, M4, M6). A hung child left until a person acts ≺ its link lost
  as at a link's going off, the child asked to end once its windows pass and ended if it has not a deadline later:
  "restarted, not crashing again of course" (M6, 10-20). A kill at any time ≺ a kill only outside a termination's window
  and while no job of it is in a call: "inducing a system halt or kernel halt causes the entire node to fail. that is
  not permittable" (10-20). On "Failure across interfaces" 4 again, SIGKILL alone, whose release can follow a post by
  microseconds (the October 16-17 aborts' open trigger was a release while posts could be in flight) ≺ SIGSTOP, then
  SIGKILL once the device has been quiet MESH_PLATFORM_NS, as after a halt; on M3 the second costs one deadline more,
  and both end.
- **A child's main loop that stops while its lanes still serve** (section 5: "no transaction fails because time
  passed"; the operator, 2026-10-22: "we don't introduce race conditions or use timing anywhere"; C5's upper pole:
  "within a call, the only thing that ends a wait is the data or the platform deadline"). A child kept serving whose
  parent can no longer halt or end it (it reads no halt, tells no failed lane and answers no job: a state nothing
  ends, M3's lower pole) ≺ its link lost at the platform deadline, its calls in flight failing as values, as at a link's
  going off (C6's upper pole, "the call returns an error, and the caller runs it again"). The deadline here bounds the
  bridge's own lifecycle, as section 5 has it, not a wait on a peer's data; and by the inventory above the main loop
  has no wait that can reach it.
- **A thread in the kernel** ("Failure across interfaces" 4). A watchdog that signals, restarts or reboots to clear it ≺
  one that counts it and leaves it to the kernel, the node reachable and its other ports serving: no signal reaches it,
  a parent's restart releases no child's call and loses every other link, and a reboot is the node failing.
- **The parent's sends** (M3). A blocking send to a child that may not read: "a wait whose end depends on an event that
  may never come" ≺ a send that never waits, a full socket read as the child's hang (10-20).
- **The provider's fault** (the operator, 2026-10-22: "process isolation of 'things using hte driver' should totally and
  absolutely contain absolutely all faults or exception-like-errors that could ever relate to the dma driver itself.
  nothign we write in any other namespace has any justification for failing transitively because of a hardware
  failure"; M3; "Failure across interfaces" 4). The fault recurring in the child until its parent ended it, a job of
  it held out and with it every port's jobs (b53804bf; the review of it, finding 6) ≺ the fault absorbed in the port's
  child and ended within its own life, its parent seeing a link lost and nothing more.
- **The bridge's own bounded waits** (M3). A wait of the bridge's bounded at MESH_PLATFORM_NS while it is watched (the
  stop's single waits, a leaving child's), which its watchdog would read as a hang ≺ each bounded at MESH_PAUSE_NS, the
  progress word written between.
- **The statistic** (Quantities). A hang known from a log line ≺ counters a reader takes whole: "a measurement without
  those values ≺ one with them"; "a channel 5 s without anything happening on it ≺ one that is never so long quiet".
