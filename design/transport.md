# The transport: one owner of the device, clients of its pages

This is the description of how data moves between nodes, who may touch what, how it fails and what that costs. It
answers, in terms of the objects the code holds, the operator's questions of 2026-10-19. Where it and older text in
this repository disagree, this text is the description and the older text is history.

The 2026-10-24 exact-byte revision below is wire version **135**, region layout **154**. It passed local build,
sanitized/mock ownership tests, and the [live validation below](#exact-byte-validation-on-2026-10-24). The cluster
deployment was temporary; the preceding bridge was restored after testing. These results do not describe a permanent
deployment or establish that all workloads attain their bandwidth bounds.

Incident records it cites: [RDMA-KERNEL-RECOVERY.md](RDMA-KERNEL-RECOVERY.md) (September 5, October 16-18). Apple's
transport note: [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt).

## 1. The objects

- **The registered region.** One POSIX shared-memory object per node, `/mesh0`, mapped by the bridge and registered
  with every Thunderbolt RDMA device of the node by that port's child as it opens the device (section 14), one ibv_mr
  per extent of `MESH_EXTENT` bytes (1 GiB, the bridge's choice), aligned so that each lies inside one 4 GiB bank. The
  hardware fact is the bank: `tbt_post_recv` keeps only the low 32 bits of an SGE address with the extent's key, so an
  extent crossing a 4 GiB boundary aliased receives 4 GiB low (September 5). Every address a request names is inside
  it.
- **The page table.** The region's rows (one row is one block, 16 KiB on the pair) and, per row, an owner word and the
  arena page it maps (`mesh_buffer`, `mesh_page_entry`), claimed by compare-and-swap on bitmaps. A client's memory is
  rows whose owner word is its own. This allocation granularity is not a minimum transfer size. "Lease" in the code
  means only this: rows owned. It has no term, renewal or expiry.
- **A link.** The two ports at the ends of one cable and the queue pairs the bridges make on them at each pairing:
  the unreliable-connected queue pairs of its plan, ten where both ends' devices hold ten (macOS 26's) and fewer where
  either holds fewer (the M5 Ultra's macOS 27 devices report max_qp 3, two usable by the max_qp − 1 rule:
  [HETEROGENEOUS-LINKS.md](HETEROGENEOUS-LINKS.md)),
  its streams' lane pairs and, where the plan has them, lanes 0 and 1 (nothing posted on either)
  ([HETEROGENEOUS-LINKS.md](HETEROGENEOUS-LINKS.md) section 10);
  created, moved through INIT, RTR and RTS, on two
  completion queues of the port's device context.
- **A posted receive.** An `ibv_recv_wr` with one SGE: an address in the region, a length, a key. The queue pair's
  next arriving SEND fills the oldest posted receive. Its completion is an entry in the receive completion queue.
- **A SEND.** An `ibv_send_wr` with one SGE naming a range of the region. TN3205: only SEND and RECV exist on this
  transport (no RDMA WRITE or READ), a SEND and the receive it fills must cover the same number of 4 KiB frames, and a
  SEND is held by the hardware's credit until the peer has a receive posted. The compiled plan makes the SGE lengths
  equal exactly. `ceil(length / 4096)` is the credit charge, not padding read from or written to application memory.
- **A resolved byte span.** `mesh_dma_span {uint64_t offset, bytes}` is 16 bytes of local immutable metadata; offset
  is relative to the region base. It names the actual registered interval, can start inside an allocation page, and
  never crosses a 1 GiB MR boundary. No copy, header or descriptor travels with the payload. Physical wire framing
  overhead is separate and is not inferred by treating a credit frame as 4096 transmitted payload bytes.
- **A readiness word.** A 64-bit contiguous count of compiled span units, with one writer and one or more readers:
  `ready` counts produced units, `sent` completed SEND units, and `landed` completed receive units. Both hardware
  lanes contribute to the same contiguous prefix. These counts are neither byte counts nor allocation-page counts.
  All-ones (~0) in a completion prefix means loss, not completed work. Invalid descriptors fail validation; the
  byte-extent revision does not substitute an all-ones payload (section 10).

The compiler binds storage and dependencies; the bridge realizes exact spans, requests, completion prefixes and
queue capacity. It need not know tensor dimensions or the arithmetic performed on the bytes.

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
- the bridge's own pages (each port's link pages, its streams' lists and physical-interval pin rings, the map, and
  the reserved all-ones block retained from older layouts) are recorded in the region and taken again;
- every stream is set not live and held by none, its landed words ~0, so a communicator of the old pairing fails and is
  made again, and the next pairing opens each stream new;
- a retirement a dead process left half done is finished.

A region a live bridge serves is never touched: a second bridge of the node ends at its port's bind, before it reads
the region.

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
  A process in its exit is none of them: XNU takes it from every lookup by pid as the exit begins, before the kernel
  releases its device objects (metal-microbench docs/facts.md). On 10-23 the heartbeat so made a region beside port
  children of the reverted line that never left that release (metal-microbench
  output_data/review-20261023/LIBRARY-STATES.md, section 2).
  Fields of the header, the slots and the bridge's area are only ever appended, so the heartbeat reads the holders of
  a region of any layout.

A second region is never made beside a live holder of the first, because on the Mini (24 GB, a 17 GB region) GPU work on
one region beside another's registration wired all its memory and its watchdog halted the node (2026-10-20): the
heartbeat opens a bridge only once no expired region is named. A process that maps the region without a slot (a reader,
or a child forked by a client without exec) is not recorded; none registers it or makes it resident.

**The heartbeat** (rdma/mesh-heartbeat, the job launchd keeps alive, `io.mesh.bridge`) supervises the bridge. It holds no
device, region or socket of the mesh. It opens the bridge (a child in a process group of its own, so the bridge serves
on if the heartbeat ends), closes it on the operator's signal and hears its end as a kernel event (EVFILT_PROC
NOTE_EXIT), never inferred from a delay. It is the one process that unlinks an expired region. Its signals, sent with
`launchctl kill SIG user/$(id -u)/io.mesh.bridge`:
- SIGHUP opens the bridge, or restarts it: the bridge closes by SIGHUP (its region kept), the heartbeat runs its own
  image again (a deploy's new binaries both, bin/mesh-user.sh), and the bridge opens and adopts the region;
- SIGUSR1 releases: the bridge closes by SIGTERM, the region is unlinked once none of its holders lives, and no bridge
  opens until SIGHUP;
- SIGTERM and SIGINT (launchd's stop of the job, a shutdown) release and end the heartbeat, which exits 0, and launchd's
  KeepAlive {SuccessfulExit: false} does not start it again.

A bridge that ends unasked opens again 60 s later, so the mesh is never restarted more than once a minute (rubrics 6: "do
not stop meshes hundreds of times per second, minute, or hour"); nothing else is built for a crash (rubrics position
5). A bridge that ended finding its region expired opens again once the heartbeat has unlinked that region. A job whose
definition predates the heartbeat runs the bridge directly until the job's next load (launchd keeps a loaded
definition): its stops keep the region as ever, and it has no release.

**A client** (a rank, a tensor function, torch) owns its rows, and for each of its calls the positions it sends and
receives. It reads and writes its own rows and its readiness words. It names nothing else.

Why the cut is there: the device's state lives as long as the node and its cables (registration of most of RAM, links
paired and re-paired while no call runs, a link's queue pairs shared by every client), and a call lives for one
call and may fail, be revoked or be killed. A call that held device state would leave it half-done at its end, and
another party would have to repair it with verbs calls. No call holds any.

**Why some of the bridge's state must be unnameable by clients.** Four kinds of state are shared by parties that fail
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
   cannot hang on a party that can vanish. Only a single owner can always finish what it started.

Execution readiness uses single-writer prefixes: the client writes "supplied"; the bridge writes "sent" and
"landed". Local ownership and list-publication metadata have their own atomic protocols, independent of payload
readiness. None is an application receipt sent across the cable.

## 3. Execution has no request/ACK protocol

A protocol is needed only where one side must learn something neither side can know beforehand. Here that is the
queue-pair numbers, packet sequence numbers and GIDs the hardware assigns at each pairing (TN3205's out-of-band
endpoint exchange): HELLO and LINK on the control socket, once a pairing. RECORD carries each node's links and memory
(the map), which is data, not a dialogue.

Before execution, both ends have equal payload lengths and unit order. `ncclMeshWritePagesPrepare` and the
[tensor adapters](../rdma/mesh-write-compiler.h) resolve local page selections and exchange shape metadata
through the existing bootstrap while realizing the plan. `ncclMeshWriteExecute` exchanges neither that metadata
nor acknowledgments. Affine rows stay in a fixed-size report. Peers may use different row pitches, row shapes or
indexed selections for the same message. The [common cut builder](../rdma/nccl-mesh.c) includes actual layout
discontinuities, each binding occurrence's registration boundaries, publication tiles and the packet limit.
Typed repeats retain the [modular repeat compiler](../rdma/mesh-span-plan.h). Only final DMA descriptors are
materialized. Full pages and masked pages use this same compiler and executor.

Data executes as SEND into an AOT-bound, preposted receive. There is no payload header, receive-address request,
receipt all-gather or remote ACK. A lost link remains a driver/control-socket event. Prepared bindings carry a
configuration-time agreement; this is not a claim that creating a plan involves no metadata exchange.

### Exact-byte validation on 2026-10-24

The three-rank [byte-write test](../rdma/tests/byte-writes.py) passed all five cases: 3 bytes; seven 257-byte rows;
5,003 three-byte rows; 4 MiB + 13 bytes; and 1 GiB + 19 bytes crossing registration boundaries. It verified exact
payload accounting and preserved leading, trailing and inter-row neighbor guards. The larger transfers carried
payload over all six discovered cables. Separately, [fused-physical](../rdma/tests/fused-physical.py) passed 14 cases
and 94 calls per rank in each of the two-node and three-node runs.

The complete numerical function `seed -> remote uint8 affine transform -> local xor` used five trials of 128 calls
per mode. Median amortized function latency was **31.290 µs with exact spans**, versus **35.569 µs with padded
transfers**. Each transfer's payload was 3 bytes versus 16,384 bytes. These are whole-function GPU observations,
including computation and communication; they are not measurements of isolated NIC latency, nor does the payload
ratio imply the same latency ratio. Preparation, compilation, allocation and diagnostics were outside timing.

The subsystem improvement is established by preserved results with fewer payload bytes and removal of the remote
packing/unpacking passes. It does not require a visible speedup in every complete function. Lower interconnect
overhead frees resources that dependent computations can use; the observed latency effect depends on how much
that overhead contributes to the workload. Whole-function measurements characterize that effect and check for
regressions. In the separate large-DNN replay every observed span was already 16 KiB, so it exercised the revised
transport and accounting without exercising the short-span padding savings.

The [validation logs](/Users/mdot/metal-microbench/output_data/review-20261024/byte-extents) record this temporary
wire-135/layout-154 deployment and its restoration. The historical measurements elsewhere in this document remain
evidence for their respective revisions.

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
- a port's progress thread (`ibv_poll_cq`, which returns at once, and reads of supplied words);
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

An expiry stops the process's GPU work before the program ends (rdma/nccl-mesh.c mesh_library_expired): it takes every
communicator's failure claim and stop, so each GPU wait and the track kernel end within 2^10 loads, and awaits the
library's committed GPU programs within MESH_PLATFORM_NS. A watch looks once a second for a communicator's command
buffer committed and not ended while none of the library's programs has ended for twice MESH_PLATFORM_NS, and ends the
program so, naming the wait.

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
   each port's progress thread sees it at its next pass and posts nothing more, it sends FAREWELL on every paired port, and
   it exits. It makes no verbs call. Whether the Mini finishes shutting down within the second does not matter: if
   power goes first, its kernel state goes with it, and a machine that is off cannot hang.
2. **The node staying up** (the M5). It reads FAREWELL, or the link-off event, or the peer interface's termination.
   It stops serving that link and makes no verbs call on it. The termination finishes because nothing of the bridge
   holds the provider's gate. Calls with positions on that link observe failure metadata; the byte-span path does
   not fabricate replacement payloads (section 10). Other ports continue independently.

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
  with no destroy calls it has not been seen. Where it happens, the post's thread faults, and the port is lost, as
  the cable is.

Positions in flight on the crushed link are transport errors (section 11). Nothing needs administration.

## 9. Should the bridge ever stop on its own?

No. It does not stop because of a peer, a link, a client, a call, a time or a fault. A stop exists only to replace
its binary, or to release the node's memory, and it is started from outside (its heartbeat's SIGHUP, or its SIGTERM:
section 2). The stop is: a flag; the port threads see it and
post nothing more; FAREWELL on every paired port; exit. It makes no verbs call, needs no interrupt handler (the signal
handler only sets the flag), and cannot enter the October 18 deadlock, which needs a deallocation call in flight.

The one exposure left is the kernel's own release of the device at the process's exit coinciding with a termination,
which is what upgrades avoid by their timing: one node at a time, with no program running and the peer's bridge
steady.

Nothing of the node's brings a fabric port down: a port whose cable link is up with no RDMA peer is logged and held
(`bin/mesh-fabric-init.sh`), since a port taken down under the bridge's open device terminates the device beneath its
objects. A job an operator disables (`launchctl disable`) stays off: the installers load no disabled job
(`bin/mesh-launchd-agent.sh`).

## 10. Invalid descriptors and historical duds

Earlier page-only versions could replace an invalid row with a whole page of all-ones, sent from the bridge's
preinitialized block. That history does not describe the byte-span path. `mesh_stream_present` validates the span
table, dependency word, arena bounds and exact registration-contained extents before publication. A bad span seen
by the bridge fails the link; it is not converted into filler or a padded write to unrelated storage.

Previously posted receives retain their actual destination intervals until completion or the posting child's
queue pairs cease to exist. Failure and an all-ones completion prefix never prove those accesses completed.

## 11. Reliability: three nines

The operator (2026-10-19, verbatim): "i am now demanding that there be 3 nines of reliability between bridge or
transport level errors. you need to transmit 1000 frames which are *normal* and have no errors in their transmission
for every 1 *error* wrt the bridge application. sending an all-1 dud buffer is not a transport error, but a client
error".

- **A frame**, for the accounting below, is a hardware credit unit charged to a posted SEND. A short span consumes
  one credit unit without transferring 4 KiB of application data. Physical framing bytes require separate measurement.
- **A normal frame** is a credit unit of a payload that arrives intact in the receive the schedule gave it.
- **A transport error** is a frame that does not: lost, cut short, landed in another position's receive, held by the
  bridge past its deadline, or cancelled because a link was lost or concluded. Every frame in flight on a link when it
  is lost counts, and so does every frame scheduled on it until the calls on it end.
- **Not transport errors:** the client's historical dud substitutions, and positions of calls that ended before
  they were sent. The byte-span path introduces no dud substitutions.
- **The requirement:** at least 1,000 normal frames per transport error, counted per bridge.

What it rules out: any link loss the bridge causes itself. One link loss in a decode call at the pair's rate (on the
order of ten thousand frames a second) costs every frame scheduled until the call ends; a re-pair of about 8 s
costs tens of thousands, which then requires tens of millions of clean frames to stay at three nines. Before this
document the bridge concluded a link at every rank renewal and at every failed call (section 12), which no workload
can meet. After it, only physical events (a cable pulled or crushed, a node losing power) lose links.

Keep posted and landed credit-frame counts, actual payload bytes, and transport-error counts distinct in each
node's record. Historical dud counts belong to the versions that emitted them; they are not useful payload or an
exact-byte failure recovery mechanism.

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
| Where receives go | into a communicator's rows, restarted per communicator (HEAD before 127) ≺ into the stream's in ring, posted by the bridge for the link's life, which clients copy out of (127-131) ≺ into the holder's own landing pages as it presents them, the bridge's spare page between holdings (132) ≺ into the holder's own landing pages, nothing posted between holdings (133) | M4, M9, M8 |
| How a holding ends | the link concluded ≺ every unit it declared or supplied sent, the unsupplied ones as duds, and one more (132) ≺ at once at its give: nothing more of it posted, the stream free (133) ≺ at once at its give or its holder's retirement, each unit told before it is posted and posted only while the holding is its holder's: with nothing outstanding the stream free, else the stream not live for its pairing (134) | M4, M10 |
| Where the next holding starts | the bridge's answer in the client's slot (127) ≺ each lane's counts sampled as a holding begins, exchanged as `start` words and MAXed, spares and duds between (129-132) ≺ where the last one ended at each end, the stream's pages posted told as facts, the filler presented by the collective (133) ≺ where the last one ended, equal at both ends of a live stream by construction, the facts kept as a statistic (134) | M2, C7 |
| A holding that ends with units outstanding | its receives' owner's every row held until they land (`touching`), the next communicator's filler and its wait taking them up (133) ≺ the queue pairs moved to Error and every request flushed (the operator's permission of 10-22, not used) ≺ the stream not live for its pairing, a link with no live stream lost by the port child's exit, the window's rows held until each unit completes or the child exits (134) | M3, M4 |
| Which rows a reclaim keeps | every row of a holder a stream still names (`mesh_touched`, 133: D(P) taken as everything P held) ≺ W(t), the rows the units posted and not completed name, the plan's window in flight, each until its unit completes or its port's child exits (134; design/extents-and-liveness.md, d) | M4 |
| A queue pair's re-pairing | moved to RESET (127, panicked the M5) ≺ destroyed and made again ≺ none: made and connected once in the bridge's life, a lost link down until a person relaunched it (129) ≺ none in a process: made and connected once in a port child's life, released by its exit, the link paired again by the port's next child (130) | M3, M4, M6 |
| A progress thread with nothing to do | parked on a bell clients ring (127) ≺ spinning while the link serves, waiting on the guard word while it serves nothing | M8 |
| The binary changing | the bridge stops itself ≺ it stops on its heartbeat's SIGHUP alone | M3 |

**The boundary (version 132; the operator, 10-21).** "the bridge, which is an api for controlling DMA, and the collective
clients and data processing code, which present buffers for transmission and accept whatever packet or frame size the
bridge chooses to use. the bridge chooses, and it chooses based upoin the actual driver and api the rdma firmware
supplies. only the bridge is allowed to know these configuration details." And: "zero copy methods must always be used
over access by copy because memory bandwidth and flops are too scarce on this hardware api." Versions 127 to 131 had
clients copy into the bridge's rings and count positions against them (a lag credit), so clients knew frames and units;
the fix of version 125's real defect (clients holding work requests and queue pairs) had moved the data itself into the
bridge. A client naming its own pages holds no DMA state, and needs no ring.

**The stream (wire 135, layout 154).**
- Each physical cable carries up to `MESH_LINK_STREAMS` streams, according to its negotiated queue-pair budget.
  Unit p goes on lane p mod 2 as unit p div 2 there; both ends post equal exact payload lengths in this order.
- Each stream has a `wire` identity derived from its pairing. Local holding generations distinguish later users
  of the same pairing and its reset completion words.
- A holding presents SEND and landing lists of `mesh_entry {spans, units, word, base}`. `spans` is a region offset
  of an immutable `mesh_dma_span {offset, bytes}` array; `units` is its count. A table can describe contiguous,
  unaligned or scattered intervals and arbitrarily many affine rows. Descriptor count follows actual packet cuts,
  not one host call or one list entry per row. List entries are 32 bytes; span records are 16 bytes, all local metadata.
- The compiled plan gives both endpoints matching units. Each SGE names its exact byte interval with the proper
  1 GiB registration key; payload does not round to an allocation page or hardware credit frame. Parallel paths
  and relay buffers use the same descriptor. Storage ownership must hold for every physical reader.
- A SEND unit k is available when its selected word (0 selects `ready`) reaches `base+k+1`. A receive's optional
  word permits first write at `base`; fresh disjoint destinations need no reciprocal publication. `landed` advances
  only through the completed prefix across both lanes, enabling the actual dependent consumer without a receipt.
- Queue accounting retains two separate limits: `ceil(actual_length / 4096)` frames per request, and the outstanding
  request slots in the `MESH_LANE_UNITS` pin ring. Receive completions retire their actual cumulative frame charge.
  Frame capacity is not permission to send padded application bytes.
- Before a post, the bridge records its actual physical `{offset, bytes}` pin and slot generation, publishes the
  posted count, and rechecks the holding. Reclamation intersects that interval with every backing allocation run;
  logical row IDs or alias tables cannot hide an overlap. The historical stream field `rows` now locates interval
  pin records, not a single row ID per DMA unit. A racing snapshot conservatively retains storage.
- `taken` releases a list slot, not the immutable array named by it. Tables remain live through every future use
  and final SEND/LAND completion. On give, their CPU readers require the bridge's subsequent owner transition to
  0 or GIVEN-alone before incomplete table uses can detach. Payload request pins do not cover CPU metadata reads.
- Abrupt client exit uses a separate bridge CPU-read pin. Before dereferencing a table or external dependency word,
  the progress thread publishes those exact intervals and then rechecks the holding/client with sequentially
  consistent atomics. Retirement either observes that pin or wins the recheck, preventing the read. A monotonic
  sequence makes a torn snapshot conservative; clearing the scope or the posting child's exit ends the CPU reads.
  This retains table/word storage without keeping unrelated allocations and without any remote acknowledgment.
  The pin occupies unused stream padding; existing field offsets, span descriptors and the 384-byte stream ABI stay
  unchanged. The sequence does not reset at a new holding or pairing.

Diagnostic span-length history has a separate lifetime from the DMA descriptor table. Active trace observers
retain the lengths needed to reconstruct their windows, even after completed DMA descriptors retire. Without an
observer, completed history is pruned under the native group lock, retaining cumulative unit/byte baselines and
the readiness checkpoint. Trace stop joins its observer before releasing history; overlapping traces keep it until
the last observer ends. This bookkeeping neither retains payload bytes nor sends a receipt.

The bridge's HELLO still requires equal allocation-block sizes because generic collective front ends retain a
common packet grid. Exact byte DMA itself imposes no such equality. Remove this remaining compatibility check
only after every front end prepares matching cuts for unequal block sizes.

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
- Two invariants. (1) A stream's lanes carry its holders' declared byte spans and nothing else: no keepalive, no spare
  receive between holdings, no dud of the bridge's own. (2) Both ends' lanes count from 0 when the port's child makes
  its queue pairs, and each end posts the stream's units in the sequence's order. So a SEND's n-th unit on a lane
  fills the peer's n-th receive there (TN3205), and a payload's place is a function of the two ends' entries alone.
  Posting in order keeps both lanes' counts a function of one number, the stream's units posted: two lanes posted
  apart could leave a holding that ended at counts that no sequence of units fills.
- A process holds a stream as it holds rows: it takes the owner word from 0 to its own by one compare-and-swap
  (`mesh_stream_take`), nothing awaited. The port's progress thread then begins the holding, its send walk at the
  stream's units sent so far and its landing walk at the stream's units posted so far, in one pass: no unit of an
  earlier holding is outstanding then (a holding's end, below).
- The facts. The bridge tells each of a stream's two lanes' units over its pairing, each way, posted and completed
  (`posts`), and the physical interval each unit names (`rows`, the historical field name), for release of backing storage (mesh-dataflow.c mesh_pinned); no
  communicator reads them. On a live stream both ends' are equal by construction: a holding that ends with nothing
  outstanding at both ends left every SEND completed into a posted receive (TN3205) and every receive landed from a
  SEND, and one that ends otherwise takes its stream out of the pairing. The pages posted that communicators once
  exchanged at their init and kept as a statistic read by nothing (version 134's `differ`) are deleted. The filler of
  version 133 (dud SENDs and receives into the spare page presented first, entries of
  `MESH_NO_ROW`, and the init's wait for an earlier holding's receives to land) is deleted; the operator, 2026-10-22,
  on waiting for a future communicator's filler: "dispose of".
- One communicator holds a particular physical stream at a time. It can bind streams on every available cable
  between a node pair, with a matching free `wire` at both ends of each cable. Sharing a peer does not collapse
  those bindings to one cable. A binding is refused (EBUSY) when its physical link has no stream free at both ends.
- A SEND is posted when its compiled producer prefix permits it. A receive is preposted once its declared
  first-write dependency and queue capacity permit it; fresh disjoint storage needs no producer gate.
  Both are data, and neither has a deadline.

**A holding's end** (version 134; the operator, 2026-10-22: "if we have a graph serializaiton that we know tells us
that specific buffers used by a specific lcient have no future writes targeting them they can be freed earlier? ...
some algorithm tries to assume that all prorgams have arbitrary access patterns and communication schedules for the
sake of breadth, but in so doing relaxes the problem socpe they are working on so severely that they describe a less
stable or worse-converging algorithm"; on `mesh_touched` pinning every row of a dead client: "this is a claude subagent
excess and must be corrected"). The analysis, with its literature, is design/extents-and-liveness.md, in its terms:
D(P) the extents a program declares, U(P, t) those it uses, L(P, t) those some later event of it may still use, and
W(t) those requests already handed to the hardware reference, the plan's window in flight.
- The holder gives the stream by setting bit 63 of its owner word (`MESH_STREAM_GIVEN`), and the event loop does so for
  a holder gone (its exit, the event loop's EVFILT_PROC: the death is that event, not a poll). The progress thread ends the
  holding at once at the give, or at its client's retirement: nothing more of it is posted, and its entries taken and
  not posted are dropped. Each unit is told before it is posted (its stable physical interval in its lane's
  ring, then its lane's count, sequentially consistent) and posted only while its holding is still its holder's (its
  owner word not given, its client's slot its own: mesh-link.c `unit_told`). A client that gives or retires and then
  reads the lanes' units therefore sees every unit that will ever be posted for the holding. Nothing in the bridge
  closes a holding with its peer: no message, no totals, no clock.
- On its next pass, that pass's completions taken, the progress thread reads the stream's two lanes. Where every SEND posted has
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
  is so is not lost by this rule while each end keeps one live (open, for the operator: the peer is told nothing).
- Backing storage. A posted request pins every allocation intersecting its actual interval, including spans
  crossing row boundaries or resolved from aliases/scatter tables. A returned lease with any such overlap stays
  closed until the requests complete or their posting child's queue pairs cease to exist. Link loss alone is not
  completion. The stream records that child's pid and start, so reused process IDs do not supply retirement proof.
  Future unposted reads are also live uses: native `ncclMemFree` is a logical asynchronous free and retains all
  registered future physical readers before presentation. Its allocation pool separately needs completed GPU uses
  and destination-consumer proofs before overwrite. For a dead payload, giving every relevant holding permits
  outstanding requests to retain it through bridge pins; this is not an application overwrite proof.
- Immutable descriptor tables have local CPU readers in addition to their SEND/LAND direction obligations.
  `client|GIVEN` prevents a future hardware post but does not itself prove the progress thread stopped reading the
  table. Until the bridge transitions that owner to 0 or GIVEN-alone, descriptor storage remains retained. This
  observation is asynchronous local ownership bookkeeping and introduces no remote ACK.

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
  declared and changed only where the contract changes, never by comparing builds: one source builds different programs on
  different machines), in a
  process group of its own, its parent's socket at descriptor 3. It opens the port's device, registers the region,
  makes and connects the link's queue pairs (its plan's, from the queue pairs both ends' devices hold, each end's told
  in its HELLO: [HETEROGENEOUS-LINKS.md](HETEROGENEOUS-LINKS.md)) and runs the port's progress thread (one: its SENDs and receives in one pass). The parent asks it to open,
  make, connect, halt and end; it answers each, naming the pairing it was asked for, and tells failed lanes, absorbed
  faults, guards not held and terminations it heard. Queue pairs exist only in a child, and a child is the bridge
  ("the bridge and only the bridge may ever have access to that exact data structure", 10-20).
- A link's queue pairs are made once and connected once in a child's life. None is moved to another state,
  destroyed or forgotten; the kernel releases them at the child's exit. A link takes every queue pair its device holds
  for it (ten of a macOS 26 device's ten; two of the M5 Ultra's macOS 27 device's two by max_qp − 1), so a returning peer's lanes
  cannot exist beside the old ones, and the old child's exit is their one release (RESET panicked the M5 on 10-19;
  destroy is ruled out by its history, M3).
- A link lost once its child was asked for lanes (a farewell, the control socket's end, a failed completion, no stream
  of the link live, the link going off, a termination, an absorbed fault, a protocol error, the child's own end): its
  streams cancelled (landed
  words ~0), its lanes served no more (the child halted, before the control socket closes), the port closed. Once the
  port's quiet window and any termination's have ended and no call of the port's own is stuck (B9, B9b, B12), the
  parent asks the child to end, reaps it and spawns the next, which opens the port and pairs with the peer whenever it answers. Nobody acts.
  Other ports' children and links, the clients and their leases are untouched.
- A child that ends unbidden loses its link the same way and is spawned again once the port's quiet window has
  passed, at the events that open any port (its child's end is one). Nothing the bridge does next depends on how often
  a child ended: the spawn backoff of 1f9be1d5 and 128d0e84 (the k-th unbidden end within 60 s of a spawn holding the
  next 60·2^(k−2) s) is deleted (4dc9e98a; the operator, 2026-10-23 01:03Z: "this is not a real feature. delete it and
  anything like it"). A child whose parent is gone halts its
  lanes and exits MESH_PLATFORM_NS later, its own windows held; a port's child is spawned only while no other child of
  its device lives (one of the launch before, whose parent ended unbidden; one in its exit is not seen), so a device's
  queue pairs are one child's at a time and no make meets the device's ceiling. At the bridge's stop every child is halted before the
  farewell, asked to end after the exit's hold, and awaited up to MESH_PLATFORM_NS. A message to a child, or a child's
  to its parent, goes whole within MESH_PLATFORM_NS of its send, and a peer that takes none by then ends the sending
  process at once, the wait named (the operator, 2026-10-23, of the blocking send: "forbidden. no wait may ever run
  longer than a deadline operand presented at the time the wait was constructed; the presumed default is 5 seconds").
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
  the next time its turn waits for (a quiet window's end, the exit window, a hold, an open's retry, a hello, a job's
  bound, the liveness turn), at most MESH_PAUSE_NS; every other change it takes wakes it where the change is made (a
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
