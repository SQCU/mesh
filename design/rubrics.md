# Three rubrics: mesh, collective, data

## Three kinds of source

The mesh's code is three kinds of source: mesh (M), collective (C) and data (D). The rubrics below rank some of the same
comparisons in opposite directions in different kinds (Inversions). For example:
- Surviving another party's failure ranks higher in M and lower in C and D.
- Specializing to one model ranks higher in D and lower in C and M.
- Members knowing each other's parts ranks higher in C and lower in M and D.

So no piece of code can rank high in two kinds at once. That is why the three are kept as separate bodies of source,
with separate namespaces and separate interfaces:

| Interface | What crosses it |
|---|---|
| M and C | rows a client owns, positions in them, supplied and landed words |
| C and D | the NCCL API, the device-function header, memory through `ncclMemAlloc` |
| M and D | nothing |

Kept apart, each kind is simpler, fails in fewer ways and runs faster than a mixture would.

**What the bridge does (M), one process per node.**
- At its launch, it maps the node's window and registers it once on every RDMA device.
- When it first hears a peer on a cable, it makes that link's queue pairs and exchanges their numbers once. It connects them once in its life. That exchange, and the map records it floods, are its only messages.
- On each link it keeps four streams toward the peer node, each on a lane pair of its own and named by an identity both bridges compute alike:
  - It posts receives into the stream's in ring for the link's life.
  - For each unit the holder declared, it posts a SEND when the unit's ready word is set. If the deadline passes or the holding ends first, it posts the same SEND from its all-ones block instead.
  - It raises each landed word in order.
  - When the link is lost, it writes ~0 into the landed words.
- It retires the slot of a client whose process ended: it releases the client's rows and lets go of its stream.
- It ends when a signal arrives, and its exit releases everything it holds of the devices. Its window outlives it, and
  the next bridge adopts it, its clients' leases standing; a release is its heartbeat's: once the bridge has ended, the
  window expires and is unlinked once none of its holders lives, so its memory returns to the system
  (design/transport.md section 2, the heartbeat).
- It never reads or writes a payload byte.

**What the bridge cannot do: no API exists for it.**
- Answer a request: it reads words and writes words.
- Know a communicator, rank, call, collective, schedule, position or route.
- Compute, combine or copy data.
- Deliver a call or take a command.
- Re-pair, reset, flush or destroy a queue pair, or declare a quiet peer lost.

**What a client does (C), libnccl-mesh in its process.**
- Attaches one slot of its node's region.
- Takes rows of the window by compare-and-swap on the shared bitmaps (`ncclMemAlloc`).
- Reads which nodes its bridge reaches.
- Meets its peers through NCCL's bootstrap, on its own sockets.
- Compiles the same schedule as every member.
- Holds a stream toward each member's node, the one whose identity is free at both ends, by taking its owner word while the stream is free, and tells its peers where its units start.
- Lays positions as runs of stream units and declares them.
- Writes units and their ready words, and reads landed words and the in ring.
- Combines on the host or the GPU, and paces itself by NCCL's head credit.
- Ends a communicator by destroy, which gives back its stream. A revoke, NCCL's, stops its groups and keeps the stream until the destroy.

**What a client cannot reach: not in its headers or its link.**
- Ports, links, cables, lanes, queue pairs, pairings, completion queues, work requests, scatter entries or keys.
- No verbs call: libnccl-mesh and libmesh do not link the RDMA library, and mesh.h does not include its header.
- A request to the bridge, or a way to wake its threads.
- Another client's rows or slot.
- The words only the bridge writes: landed, `live`, `free`, `next`.
- Its library's internals: libnccl-mesh exports NCCL's functions, their pnccl twins and its `ncclMesh` extensions, and
  nothing else.

**What data code (D) reaches:** the NCCL API, the device-function header and `ncclMemAlloc`. It reaches nothing of the bridge's or the region's.

The region is mapped whole into every client, so nothing in the hardware enforces the lists above. Only the headers enforce them, which is why the headers expose nothing else (design/transport.md section 2). The operator, 10-20: "memory safety isn ot a requirement or even allowed for our code. we don't expose dangerous source level headers, functions, ontological objects, whatever. that's the limit and the only limit we are working with here."

**For agents, subagents and reviewers:**
1. Name the kind of every file a change touches, from the table under Quantities.
2. Rank the change against the form it replaces, using that kind's dimensions only, each dimension separately. Give the
   ranking as a pair, `old ≺ new` or `new ≺ old`, with the code that decides it.
3. Read a change that ranks higher in one kind only by taking in another kind's concern as misplaced. It is split along
   the interfaces above.
4. When two requirements conflict, decide by the rubric of the kind the code belongs to. In a conflict between kinds,
   decide by the interface: what crosses it, and nothing else.
5. Rank a rule, a memory note or a pass/fail check written in place of a ranking below the ranking.

## How to read them

Each dimension is a ranking. Given two forms of the same code (two commits, two designs, or two call paths that do the
same job), the dimension says which ranks higher. It gives no score. The poles are the limits of the ranking: no real
code sits at either one, and between any two forms there may be a third. The poles are not neutral. The lower pole is a
specific way of failing the code's dependants, and the upper pole a specific kind of success.

The standards are literal statements about code. Where the operator's words give both poles of a dimension, they are
quoted as a pair. Where they give only one, neither is quoted, so that no dimension leans on rhetoric at one end.

`A ≺ B` means A ranks below B. Every ranked pair is the operator's own judgement in this project, dated (UTC). The
quotations are the operator's, from this project's sessions.

Each rubric has the two kinds of dimension the operator scores separately: abstract ones that apply to the whole layer,
and a few fine-grained ones about particular mechanisms. Some comparisons rank in opposite directions in different
layers; see Inversions.

**Quantities.** The values the rankings compare, each entering them pairwise like any other dimension.
- Throughput in tok/s, per stream and aggregate. A measurement is ranked like any form, pairwise, by what it
  contains. The operator, 10-20, verbatim: "the rubric contains pairwise comparable qualitative judgemetns that can
  compare any two things. therefore we cannot use that notion [a rule that every change be measured]; instead, we say
  that meausrements which do not contian those specific values are lesser than measurements which do, and that there
  is no reason or excuse for narrowness of measurement, e.g. to only record tok/sec or only kl divergence, and that any
  measurement which only records kl divergence is always worse than a measurement which should have contained a
  measurement of kl div, but instead only measured aggregate throughpout paretos." The values: aggregate tok/s, tok/s
  per stream, step latency, overhead, and bandwidth per stream against aggregate bandwidth; throughput per stream and
  aggregate trade along a Pareto curve, each backend (the engine, LiteRT, a hybrid) its own where they differ in any
  sense. Earlier words of 10-20: "avoid reporting on kl div when we are not also reporting on aggregate throughput and
  throughput per stream (which forms a tradeoff w/ a clear pareto curve if we have measured anything at all, with
  different backends serving different pareto curves where they differ in any sense)"; "if you were logging the
  throughput, latency, overhead, and per-stream bandwidth vs aggregate bandwdith in tokens/s this would have been
  reported and unwritten as an obvious regression, brittleness, and deviation from acceptable and satisfactory
  preexisting code immediately". Ranked:
  - a measurement without those values ≺ one with them;
  - a narrow measurement (tok/s alone, or KL alone) ≺ a wide one, for no reason or excuse;
  - a measurement of KL alone ≺ one of aggregate throughput Paretos alone that lacks the KL it should have had;
  - a regression patched ≺ a regression unwritten (metal-microbench d673284, 10-19: "d673 will not be accepted adn
    does not need to be reviewed");
  - a description of one device's performance, for any reason, ≺ the mesh collective's: "if we are programming MESH
    COLLECTIVE OPS we are measuring the MESH COLLECTIVE therefore something has GONE WRONG if you are trying to
    describe SINGLE DEVICE PERFORMANCE in any capacity for any reason at any time" (10-21);
  - a run set aside because one member was busy, or two runs' times compared to clear code ≺ the run read for what
    the balancer did and what the collective's output did: "you aren't allowed to compare times and the load balancing
    and SECOND ENTIRE SOC AND CPU AND GPU exist SPECIFICALLY TO INCREASE OUTPUT AGGREGATE PERFORMANCE IN ALL
    CIRCUMSTANCES" (10-21).
- KL at the unembedding, against LiteRT's pair KL. While shapes work is unfinished: an export of more bytes at a lower
  KL ≺ the export of fewest bytes within a KL of 2.0.
- Two forms whose times differ by under 3% rank equal on time.
- Latency per hop: more than three times the hardware's ≺ within it (M8).
- A DMA or network channel's silence: a channel 5 s without anything happening on it ≺ one that is never so long
  quiet.
- Errors: a transport erring more than once in 1,000 normal frames ≺ one erring less; a collective client erring more
  than once in 10⁵ transmits ≺ one erring less; a dud read where its producer lives ≺ none. The operator, 10-20:
  "collective clients can only have an error one in 10000 (yes, 1e5) transmits'; duds *should not happen*." A dud read
  is a read of memory its producer was not yet allowed to write, caught by the bridge.
- Testing and investigation: more than about 12 s of testing an hour ≺ less; an investigation of more than 10 s of
  intensive work ≺ one of less.
- A collective of one member ≺ one of two or more (D9).
- Switching to a backend after its first build: more than 1 s ≺ less (D6).

| | Mesh | Collective | Data |
|---|---|---|---|
| Code | `rdma/mesh-flow` and what it publishes; the node services | libnccl-mesh, the schedule compiler (mesh-collective.c, mesh-plan.h), the device functions in mesh.metal, the bindings (torch-mesh backend.py, mesh_mpi.py) | everything that computes values (metal-microbench's engine, LiteRT's program, mesh_rank.m, rank.swift, goal_a.swift, torch programs, examples/, torch-mesh's kernels) and the configuration it loads |
| Lives | as long as the node | as long as a communicator, in its client's process | one call; its configuration, one model load |

## M. Mesh code

### Abstract

**M1. The bridge's scope.**
- Ranks higher: the form whose bridge does fewer kinds of work. The kinds it keeps are handing out registered memory and
  moving bytes between it on paired links.
- Lower pole: a general service. It delivers calls, takes remote commands, carries events and sessions, is started per
  call, and holds any feature a client found convenient.
- Upper pole: a transport that behaves like a system call. It has a window, links, SEND and RECV, and does nothing a
  client could do for itself.
- The operator's pair: lower, "the 'bridge' is not a grab bag of arbitrary ontologies anyone wants to put anywhere"
  (10-19); upper, "the bridge is there to behave like a system call all operating systems should have always been
  giving us" (10-15).
- Ranked:
  - grid.py starting and stopping both bridges on every call ≺ a resident bridge (10-14);
  - call delivery and "remote control" through the bridge ≺ "a client which runs very fast when the right buffers have
    just appeared asynchronously in memory" (10-19).
- Also: "almost no featue or responsibility could ever be assigned to the bridge service from thereon" (10-19); "half
  of our code is imitating nccl (easy!) and the other half is implementing syscall-like-apis" (10-14).

**M2. Messages about messages.**
- Ranks higher: the form in which less of what crosses a link describes other traffic. The signal that data arrived is
  the hardware's completion.
- Lower pole: a second protocol on top of RDMA. Requests, permissions, keyword states, an END on a TCP socket,
  channel messages, a header telling the receiver what to post.
- Upper pole: SEND into a posted RECV. The only exchange that is not data is the pairing of a new cable.
- The operator's pair: lower, "if you have successfully 'asked permission' to send data"; upper, "you could have sent
  the data instead" (10-19).
- Ranked:
  - the closer's END on the TCP control socket ≺ the completion on the transfer's own queue pair (10-18);
  - receives on shared lanes driven by a header ≺ a pre-posted exclusive receive (10-17);
  - seven CHANNEL message kinds ≺ none (10-19);
  - `READY`-style keyword states ≺ none (10-18).

**M3. States and their exits.**
- Ranks higher: the form in which more of the reachable states have a transition to off, taken on a signal from
  outside, and which has fewer states that exist only to wait.
- Lower pole: states that wait forever. A thread spins with no semaphore, a drain waits for markers never promised, a
  teardown call waits on a device's termination, and a process is made unkillable.
- Upper pole: from every state, a signal ends the process and releases its resources.
- The operator's pair: lower, "states which exist purely to cause deadlocks" (10-18); upper, "imagine a number for every
  isngle state the bridge cna be in. for all numbers there must be a signal whic hcan be sent to the bridge which causes
  the bridge to shut down as an application and release all resources" (10-18).
- Ranked:
  - "a singlethreaded application which chooses to 'spin forever' with no semaphore and no secondary thread" ≺ loops
    that poll and read a stop flag (10-18);
  - a kill guard keeping processes alive ≺ processes any signal ends (10-20);
  - a queue pair moved to RESET or ERR, or destroyed, inside the running process ≺ queue pairs made and connected once
    and released only by the process's exit (10-20): "why do you think you're 'allowed' to do 'teardowns' in the
    operation of any code we are discussing?"; "every single time a claudemade code artifact has tried to introduce
    'revocation' or 'destruction' it has resulted in a null pointer access and an unscheduled reboot, or an infinite
    loop and a watchdog OS restart" (10-20);
  - a wait whose end depends on an event that may never come (a poll or socket read with no timeout, a refusal only
    another party's action lifts) ≺ every wait ending in finite time: "it is your own responsibility to write code which
    doesn't stallblock unboundedly. semaphores allow all waits in all programs to termiante in finite time, e.g. 0.1ms
    or 5s. it is your own choice to not use semaphores in potentially infinite waits, a hcoice which the literal rubric
    was written to emphasize as an exercise of decision, and a poor discretion wherever it happens" (10-20).  A wait
    that spins and reads a stop flag ends when the flag is set (M8, C4 rank its parking lower); what this ranks lowest
    is a wait nothing can end.

**M4. Containment.**
- Ranks higher: the form in which a failure's effects stay with the party that failed, and in which no mechanism is added
  to survive someone else's failure.
- Lower pole: contagion. One client's end concludes a link for every client on the cable, and a node stops itself in
  the middle of DMA. A second failure (a guard, a detector, a kept state) is added to compensate for a first one assumed
  certain.
- Upper pole: a failed client's positions go out as duds and its rows are released; nothing else changes.
- The operator's pair: lower, "a second critical system failure is required to compensate" (10-20); upper, "marking
  them as overwritable at the moment it is known they have no further consumers" (10-19).
- Ranked:
  - concluding the link to clear receives a communicator left posted ≺ receiving into the failed call's pages and
    releasing them (10-19);
  - a timer that declares a quiet peer lost and fails the program ≺ no such detector (10-18);
  - a dud read as a sentinel, its all-ones bytes failing the consumer's group ≺ a dud landed as bytes like any other:
    "duds aren't a protocol and are there to prevent the limit case of 'send must happen, remote client doesn't exist
    anymore, omething has to be placed there'. type inferences are not required and you do'nt need to do anything
    here. 'duds' are not sentinel values." (10-20);
  - a node shutting down for an unknown reason during DMA ≺ a node that stays up (10-15);
  - a layer whose user can, by an action or by doing nothing, leave one of the layer's resources unrecoverable (a stream
    a lost pairing's holder keeps until it lets go; communicators refused while the process keeps a lease of a bridge
    that stopped) ≺ a layer that recovers every resource whatever its users do: "userse of a lower level driver may
    never by action or inaction cause a resource to be unrecoverable" (10-20).
- Also: "attempts to add statefulness to functions that don't need it, and in so doing create contagious brittleness"
  (10-18).

**M5. Mediation.**
- Ranks higher: the form in which clients can name less of the device and the bridge decides more of what goes where.
- Lower pole: clients hold work requests, queue-pair pointers and the addresses of the bridge's words, and claim queue
  pairs for themselves.
- Upper pole: clients see their own rows and readiness words. "the bridge and only the bridge decides what buffer write
  is sent to what buffer receive" (10-19).
- The operator's pair: lower, "clients that have direct access to the dma abi"; upper, "mediated access through a
  queue or other abstraction which allows a nil-send at the expected deadline" (10-19).
- Ranked:
  - a dud copied into a pre-posted address ≺ a dud sent from the bridge's own all-ones rows by address translation
    (10-19);
  - one namespace for clients and bridge ≺ separate namespaces, the bridge's private (10-19);
  - a bridge with requests its clients make of it ≺ "the bridge does not have any 'verbs' at all. the rdma protocol we
    are transducing *is* a complete protocol. it's done. we're not writing a protocol." (10-20), the verbs being RDMA's
    own: "the verbs are SEND and RECV. there isn't a 'send to send' or 'send to maybe receive to maybe send to send'.
    the verbs are SEND or RECV. if you have successfully 'asked permission' to send data, you could have sent the data
    instead. this is why i keep asking why protocols are being introduced." (10-19).

**M6. Recovery without people.**
- Ranks higher: the form that gets back to serving after a loss with fewer parties involved and less downtime.
- Lower pole: recovery that needs a person. It needs sudo, developer mode, a cable pulled, a monitor and keyboard, or a
  restart of the machine.
- Upper pole: links pair again on their own, the next call is placed on the map as it is then, and a mesh can be
  recabled while it runs.
- The operator's pair: lower, "a 'have a temp worker show up and plug in the computer to do xyz intervention' security or
  trust model" (10-13); upper, "replugging a mesh network at the hardware level while it runs *with no downtime*"
  (10-18).
- Ranked:
  - keyboard and monitor on the Mini ≺ "100% remote administrable" (10-15);
  - asking for administrative actions ≺ "services always respect shutdown isgnals no matter what state" (10-18);
  - a link lost after it connected staying down until a person relaunches the bridge (version 129) ≺ the port's agent
    ending once its quiet window has passed and the next agent pairing the link again (version 130, transport.md
    section 14): "this violates some of the first mandates ever written into this codebase's documentation, which
    cover reconnection of existing bridges or mesh applications as the first basic requirement for mesh networking
    code. bridges might die, collective clients might die, data procesing clients might die, but all of these and any
    of these are not contagious to other sections in the same data flow. e.g. a bridge crashing on a computer two hops
    down should result in the bridge being restarted, not crashing again of course, and becoming available again as a
    resource for collectives with nothing but an error channel indicating that calls-in-flight failed and ought to be
    retried" (10-20); "when the machine is back it is back, and the fact that it is gone is not very meaningful"
    (10-20), "making the code work is some implicit zeroeth law that stands above all other laws" (10-20).

### Fine-grained

**M7. Memory commitment.**
- Ranks higher: the form that commits registrations earlier and changes address translation less while DMA is live.
- Lower pole: lazy registration at a program's first page, which rewrites the device's address translation in the
  middle of other clients' transfers.
- Upper pole: most of the node's memory registered once, at the bridge's start. Each client sizes its own use, and
  weights and KV sit inside the window.
- Ranked:
  - lazy registration ≺ eager registration (10-15);
  - weights and KV outside the pinned pool ≺ inside it (10-15).

**M8. The hot path.**
- Ranks higher: the form with fewer steps between a client's supplied buffer and the link.
- Lower pole: per-call probing, poll-then-park, an extra hop or copy, latency more than three times the hardware's.
- Upper pole: the queues "spinqueried as FAST AS HARDWARE THREADS ALLOW on both sides of each bridge" (10-18).
- Ranked:
  - a driver probing per call ≺ the bridge handling links at its own time scale (10-14);
  - host groups that poll for 1 ms and then park ≺ no parking (10-19);
  - anything remade per call that would be faster kept ≺ kept: "everything which isn't persistent but would be more
    performant if it was persistent is now required to be persistent. 1% overhead is not acceptable." (10-20); "if
    there is 10ns of reducible overhead anywhere in our code we reduce it to 0ns. this too is something which was meant
    to be communicated by the rubric and does not require the imposition of the user's subjectivity upon code or
    results." (10-20).

**M9. Addressing.**
- Ranks higher: the form that uses what the bridge already knows about where each buffer goes, instead of
  reconstructing it at run time.
- Lower pole: an in-order receive protocol, or a header telling the receiver what to post next.
- Upper pole: positions compiled ahead, with the queue pairs permuted so they look like concurrent streams written into
  the right slots.
- The operator's pair: lower, "an in-order recieve protocol"; upper, "multiple concurrent streams which get written into
  the right expected output slots in mapped memory" (10-18).

**M10. Attribution.**
- Ranks higher: the form whose accounts assign each lost frame to its real cause and owner.
- Lower pole: the bridge's own link conclusions booked as flaky hardware, and client lateness booked as transport error.
- Upper pole: duds counted as client errors, and links lost only to physical events, each counted.
- Also: "sending an all-1 dud buffer is not a transport error, but a client error" (10-19).

## C. Collective code

### Abstract

**C1. Provenance.**
- Ranks higher: the form closer to NCCL's API and semantics and to the published algorithm it implements.
- Lower pole: invented semantics and home-grown algorithms, with objects no reference has.
- Upper pole: the literature transcribed near verbatim, NCCL's calls as the syntax, and any deviation argued from the
  hardware's interface.
- The operator's pair: lower, "a hand-written sloplib breadboarding two cpus together" (10-18); upper, "nccl is good
  enough as a syntax to allow us to directly use the dma code" (10-19).
- Ranked:
  - "sessions", "meets" and "destructions" ≺ "buffers and basically async 'awaities' and 'maybes' wrt to buffers" (10-19);
  - "complicated logic hidden inside of classmodules" ≺ "something from the established literature" for routing
    (10-04);
  - a revoke that gives back what NCCL's keeps ≺ NCCL's revoke: "why do you want anything outside of nccl" (10-20).
- Also: "if they're really features of nccl, whatever, we can use them, that's an if and only if"; and of extensions
  the hardware needs, "it is not impossible that there are non-nccl extension functions we need for this codebase to
  perform many operations which behave slightly differently from the specific firmware on the specific rdma backend on
  this specific hardware; that is not ultimately what we are here to redact" (10-20).

**C2. Invariance.**
- Ranks higher: the form whose one implementation serves more cablings, member counts, uneven counts and dtypes without
  a branch for any of them.
- Lower pole: a two-node world, where one node plays one role, the other node plays another, and only one of them
  reduces.
- Upper pole: total over any connected map and any count of members from two; a special case turns out to be the
  general case.
- The operator's pair: lower, "'this is the ying node and that's the yang node, and only the yang node does reductions'"
  (10-02); upper, "algorithms with invariant behavior to many different superficially good sounding specializations"
  (10-04).
- Ranked:
  - an E2B path written for n:2 ≺ E2B "totally uniform with all other consumers of this api" (10-04);
  - per-strategy partitions (MoE, heads, Megatron) ≺ partition "along any tp scaling dim" (10-17).

**C3. Routing as data.**
- Ranks higher: the form in which more of the route and the collective's form is a compiled object passed as an operand,
  and less is inferred where it runs.
- Lower pole: routes chosen inside a function or kernel (shortest path picked in the call), invisible to the caller.
- Upper pole: routing tables compiled ahead of time and passed as operands. They are sampleable and log-scale, and
  adjusted rarely, between calls.
- The operator's pair: lower, "implicitly inferred and inaccessible to the outer closure scope"; upper, "a literal data
  object which is literally passed as operand to a function" (10-04).
- Ranked: `mesh_route` picking the shortest path in the call ≺ a routing table as an operand (10-04).

**C4. Asynchrony.**
- Ranks higher: the form that issues earlier and waits later, and adds fewer waits between them.
- Lower pole: barriers, pauses, syncs, guards and host round trips inserted between steps.
- Upper pole: the send at the call and the wait at the consumer, the wait only on landed data, and nothing waited on
  between a window's crossings.
- The operator's pair: lower, "innocent sounding barriers" (10-17); upper, "earlier waitless implementatinos which did
  not need to be made waitful" (10-16).
- Ranked:
  - host-path calls between recorded segments (mb 391ce57) ≺ in-stream crossings (Q5P) (10-16);
  - completing in `get_future` ≺ "send at the call, wait at wait_tensor" (10-03);
  - poll-then-park ≺ none (10-19).

**C5. When time may decide.**
- Ranks higher: the form whose time-dependent decisions are diachronic (made between calls, on known buffer depth)
  rather than synchronic (made during a call, from a timing estimate).
- Lower pole: a schedule changed during a call from a measurement, or a detector declaring a peer dead by a timer.
- Upper pole: estimators updated between calls. Within a call, the only thing that ends a wait is the data or the
  platform deadline.
- The operator's pair: lower, "synchronic temporal logic (e.g. mutating somethin during a call based on a timing
  estimate or measurement)"; upper, "diachronic temporal logic: allowed, uses known buffer depth, effectively free if
  you pipeline it" (10-18).
- Ranked: a group's units declared once with one deadline from its encode, so a live producer's later units are
  nil-sent when the group runs long or its command buffer starts late ≺ no dud for a live producer: "duds *should not
  happen*" (10-20).

**C6. Failure's reach.**
- Ranks higher: the form in which a failure ends less (the call alone) and leaves less for the next call.
- Lower pole: resumption inside a call, retries, waiting out a peer, and communicators kept across calls with state the
  next call must repair.
- Upper pole: the call returns an error, and the caller runs it again on the map as it is then.
- The operator's pair: lower, "overwrought bullshit to pretend *within-call* state is recoverable, *synchronically*";
  upper, "recover from any error *across* calls, *diachronically*" (10-01, docs/principles.md section 4).
- Ranked:
  - within-call resumption ≺ "return an error (which a caller can presumably retry against immediately)" (10-14);
  - a protocol of refusals, stuck states and errors that spread to other parties (a member's stale leases refusing the
    head's group; a handler retiring a member's programs from outside a call) ≺ a call that fails and returns its error:
    "you don't need a protocol, and you don't need ERRSTUCKs, and you don't need contagious errors. 1: you should not be
    restarting bridges thousands of times per second or minute or hour or day 2: data processing code does not need to
    be mutated mid-call based on state wrt a bridge or collective client. instead, data processing code fails and returns
    an error through an error channel (e.g. a tuple index besides the value return of a function). 3: hold higher
    standards for your code." (10-20);
  - rejection filters ≺ none: "count every rejection filter everywhere in client and data processing code. then delete
    absolutely all of them. there is no room for rejection fitlers in any collective or data processing code. you are
    afaict lapsing into defensive programming and adding positive features which induce waits, stalls, guards, hangs,
    and have no data flow or specification articulating why these states must exist or how they will with 100% certainty
    exit a guarding or waiting state." (10-20).

### Fine-grained

**C7. Symmetric knowledge.**
- Ranks higher: the form in which less is learned from peers at run time, because more is known identically at every
  member from the compiled schedule.
- Lower pole: members confer during a call, or a member decides something alone.
- Upper pole: both ends know every position a priori. Metadata goes to metadata receive buffers and tensors to tensor
  buffers, with contiguity known on both sides.
- Ranked: members deciding "adopt-or-make each on their own" ≺ "a config object asserting a different control flow"
  (10-17).

**C8. Numerical candor.**
- Ranks higher: the form that declares its precision and order, and checks payload less.
- Lower pole: bit-identity demanded across members, and checksums or digests as gates.
- Upper pole: fp32 accumulation rounded once, last-bit differences accepted.
- Also: "bit identicality of floats is not real" (10-04). Summation orders that differ between members decorrelate
  their rounding, which ranks higher than one order shared (D12).

**C9. One primitive, offered as a header.**
- Ranks higher: the form with fewer copies of each primitive, offered for others to call rather than spliced into their
  code.
- Lower pole: mirrors, templates rewritten per consumer, and transport text spliced into other programs' kernels.
- Upper pole: one header, used the same way by every consumer, whose higher-order functions take the consumers as
  operands.
- The operator's pair: lower, "a mandate to write a template 20 to 40 times"; upper, "a header or abstract function /
  higher order function which takes other functions (consumers in a general sense) as operands" (10-08).
- Ranked:
  - five mirrors ≺ "one header used the same way everywhere" (10-03);
  - a recorder pass splicing transport into LiteRT's kernels ≺ a module or header (10-14).

## D. Data code

### Abstract

**D1. The universal before the particular.**
- Ranks higher: the form whose definition serves more of linear algebra with one interface, with a model as the
  demonstration.
- Lower pole: code written for one model, one phase or one parallelism kind, such as decode-only rates or a Gemma path.
- Upper pole: any chain of matmuls, matvecs and reductions. A model's program is configuration of the general one, and
  scaling to unseen domains is "table stakes".
- The operator's pair: lower, "rotten code with a narrow cased and special mandate" (10-05); upper, "the universal comes
  before the particular and we use the particular ot show off that the universal is correct" (10-17).
- Ranked:
  - rates that measure decode stretches only ≺ rates over "an entire function call and all that it entails" (10-05);
  - TP kinds added to the Balancer one at a time ≺ "absolutely all forms of tensor parallelism should be heterogenous
    and autotunable" (10-06).

**D2. When choices are made.**
- Ranks higher: the form that makes its choices earlier: when the configuration is loaded or synthesized, rather than at
  the call, and at the call rather than inside a tensor function.
- Lower pole: parameters or ranges derived, guessed, defaulted or looked up inside functions that take tensors, with
  fallback forms when a lookup misses.
- Upper pole: every choice synthesized once at load from configuration passed as an object. A missing choice throws.
- The operator's pair: lower, "'guessing' ranges is literally by formal construction the same behavior as trying to
  access unallocated or garbage data out of the bounds of the buffers"; upper, "functions should throw or make a computer
  turn off before they try to guess parametric values" (10-19).
- Ranked:
  - a resolver's enumerated table with fallback paths ≺ no fallback (10-19);
  - argparse strings and hardcoded defaults ≺ configuration documents passed by consumers, with didactic defaults only
    (10-15);
  - configuration read as files on the hot path ≺ records from protocol, database or file alike (10-16).

**D3. Control flow.**
- Ranks higher: the form with fewer branches. Its control flow differs only where a configuration says so.
- Lower pole: escape hatches through which a function chooses its own path.
- Upper pole: as few branches as physically attainable. "why should control flow ever be different for any reason but
  'a config object asserting a different control flow was submitted'?" (10-17).

**D4. One implementation of each computation.**
- Ranks higher: the form with fewer implementations of each computation, and with foreign code called rather than
  edited.
- Lower pole: a variant layer per model part (a drafter's own transformer layer), copies of the same operator, and
  monkeypatching.
- Upper pole: one layer program run by target and drafter alike. Attention takes arbitrary mask-mod closures, and the
  mesh ops are one import.
- The operator's pair: lower, "23 different rmsnorm implementations in random places" (10-14); upper, the same mesh ops as
  "a header import and oneliner for two sorts of modules" (10-19).
- Ranked:
  - a drafter defined as a second hand-written layer ≺ the target's layer program over the drafter's weights (10-19);
  - fixed mask data ≺ "a ltieral maskfunction" compiled from a router (10-19);
  - programs split into training and inference ≺ one program queried for either (10-08).

**D5. Shares.**
- Ranks higher: the form that divides work across more axes by evidence. Unequal members get unequal work, and every
  member works.
- Lower pole: equal shares forced on unequal machines, or a member benched at 0%.
- Upper pole: every partitioned axis a ragged share from the calls' evidence. Every member hosts, and a loaded member
  gets less work, not none.
- The operator's pair: lower, "refusing to use it at all is never correct"; upper, "the *second entire computer* is a
  resource" (10-17).
- Ranked:
  - a split moving toward "M5 alone" ≺ the Mini holding "a workable chunk of ffn columns and attention call work" (10-16, 10-17);
  - a heavily loaded host producing a 0% share ≺ "MORE rows on an unloaded mac mini" (10-17).

**D6. How allocations adapt.**
- Ranks higher: the form whose allocations are revised between calls by an estimator that beats the update-free
  baseline, rather than priced by a model or looked up in a table.
- Lower pole: cost models, prices and hand-written tables (by chip, by shape), and solving each layer separately from
  partial results.
- Upper pole: the tpconfig loop. It starts from a uniform prior, times are piped out of the call, a Bayesian update runs
  concurrently, the next call gets tpconfig′, and switching is cheap enough that exploring is nearly free.
- The operator's pair: lower, "cost model inference inside of leaf closures"; upper, "iterative estimators which update
  between function calls" (10-05).
- Ranked:
  - a table of row classes ≺ no table: "we cannot table (can't ever table)" (10-16);
  - a fitted price that refused the fused kernel ≺ no price (10-17);
  - solo-node measurement ≺ evidence from the mesh's own calls (10-05);
  - per-layer oversolving ≺ coarser shares (10-17).

### Fine-grained

**D7. Mediated use of the mesh.**
- Ranks higher: the form that knows less about links, peers, routes and the bridge, and reaches the mesh only through
  the collectives API and its own node's bridge.
- Lower pole: reading the bridge's region, watching its process, probing links per call, Wi-Fi to a cabled peer.
- Upper pole: topology unknown to the program; a node asks its own bridge what neighbours it has.
- Ranked:
  - a node learning its neighbours by "sshing into a terminal mode on a mesh peer" ≺ "querying their own mesh bridge"
    (10-16);
  - Wi-Fi to a peer cabled to this node ≺ its cable: "you should never be using networking over wifi etc when it is
    physically and literally tethered / cabled to a host machine" (10-16).
- ssh and TCP: see "Two channels" (below the kinds), which ranks them; neither is struck out.

**D8. Memory the program knows it needs.**
- Ranks higher: the form whose buffers are sized by the program itself from its configuration, held from load and
  reused, caches included.
- Lower pole: ceilings computed by someone else's estimate, memory taken per call, and caches skipped "to save memory".
- Upper pole: the client knows its allocation. Everything it shares sits in the window, zero-copy, and persistent
  caches and memos are kept.
- Ranked: recomputing what a cache would hold ≺ "caching and memoization are required" (10-16).

**D9. Evidence.**
- Ranks higher: the form whose evidence is rankable statistics from end-to-end calls on two or more members, sampled at
  temperature 1, rather than pass/fail checks.
- Lower pole: pass/fail tests, greedy outputs, surrogate correctness on one layer, runs on one member, grid searches and
  long runs, and slowness blamed on other load.
- Upper pole: KL at the unembedding after the call; tok/s per stream and aggregate reported with every change; the
  history of allocations used as evidence.
- The operator's pair: lower, "pass/fail tests are generally not good"; upper, "rankable choices are broadly good;
  statistics which inform qualitative choices or decisionmaking policies are good" (10-14).
- Ranked:
  - greedy outputs ≺ temperature-1 draws (10-14);
  - a test on a collective of one ≺ a collective of two or more (10-19);
  - a 5.5-minute benchmark ≺ about 100 cycles of the kernel (10-07).

**D10. Pace.**
- Ranks higher: the form at or past the best measured frontier. Regressions are found and removed rather than
  explained.
- Lower pole: explanations for slowness ("dispatch bound", "there are crossings", "M5 alone", another program running),
  comparison against a slow reference such as Hugging Face eager, and incremental tuning.
- Upper pole: the hardware's limit reached at once. LiteRT is the bar, and a deleted fast form is restored, not
  redesigned.
- The operator's pair: lower, "justifying something running slower than it should"; upper, "no part of any model or
  eval we study may ever be slow" (10-19).
- Ranked:
  - deleting a measured form without replicating its frontier ≺ restoring it (10-17);
  - a 4x regression met by redesign ≺ the old form restored (10-16);
  - Hugging Face eager as the reference ≺ LiteRT as the bar (10-08).

**D11. Kernel units.**
- Ranks higher: the form whose kernels work in matmul tiles, fitted parametrically, and never run slower than the dense
  reference.
- Lower pole: matvec-then-give-up, autotune tables matched by exact shape, and sparse kernels that sometimes lose to
  dense.
- Upper pole: tiles with padding and masks, parametric fit by shape, sparsity measured or induced. The weight format
  serves loading, decoding and the tensor op at once.
- Ranked:
  - exact-identity autotune ≺ "parametric fit" over tile units (10-07);
  - a tile-sparse kernel slower than torch at full density ≺ "strictly faster or no slower" (10-07).

**D12. Correlation of error.**
- Ranks higher: the form whose errors are decorrelated with one another, across positions, layers, chunks of a sum,
  members' shares, forms of the same computation, and runs.
- Lower pole: errors that share a direction: one rounding direction in every chunk, a quantizer whose error aligns with
  its weights or its inputs, a model trained or measured only on data whose errors its own errors steer, and two forms
  that share a defect. They add coherently, and no average or comparison exposes them.
- Upper pole: errors with no correlation. They add in quadrature, average toward zero, and show as disagreement
  wherever two forms are compared.
- Why, in each of the operator's cases:
  - **Accumulation.** n errors of variance σ² with pairwise correlation ρ sum to variance nσ²(1 + (n − 1)ρ): about √n·σ
    in amplitude where ρ = 0, n·σ where ρ = 1. Rounding errors that behave as independent give a backward error of order
    √n·u, not n·u [Higham & Mary 2019, SIAM J. Sci. Comput. 41(5)].
  - **Quantization.** Dither and stochastic rounding make the error unbiased and independent of the signal, so it
    averages out where deterministic rounding's bias does not [Widrow & Kollár 2008, *Quantization Noise*; Gupta et al.
    2015, ICML].
  - **Floating-point non-associativity.** Summation orders that differ between members or calls decorrelate their
    rounding: last-bit differences accepted (C8). One order forced everywhere gives every member the same bias.
  - **On-policy and off-policy, training and inference.** A learner's errors on its own trajectory are correlated with
    the states those errors lead it into, and compound over a horizon T (behavior cloning's O(T²)). Data drawn from
    its own policy breaks that correlation (O(T)) [Ross, Gordon & Bagnell 2011, AISTATS, DAgger]. A measurement
    teacher-forced on a reference's tokens and one free-running on the program's own draws differ for the same reason,
    and are reported apart.
  - **Measurement.** Two forms are compared by the divergence between their own records (`tools/retained.py` on the
    two), never by the difference of their divergences from a reference. KL is asymmetric and obeys no triangle
    inequality: two forms equally far from bf16 can be identical or far apart. Decorrelated errors are what make that
    comparison see each form's error. Correlated ones hide it.
- Ranked:
  - correlated errors ≺ decorrelated errors: "decorrelatd errors are better than correlated errors for various
    reasons ... this is the case whether measuring on policy vs off policy training / inference, whether measuring
    quantization approaches, whether measuring floating point nonassociativity, etc" (10-20).
- Also, the case that prompted it (10-20, e2bheld teacher-forced over the M5 and Mini pair):
  - The 10-17 program and the 10-20 program read KL 0.616 and 0.826 from bf16, but 0.43 and 0.64 from each other.
    Their errors are decorrelated.
  - The serial walk and per-channel tracks read 6.2 × 10⁻⁷ from each other, against 3.4 × 10⁻⁵ between two runs of
    one program: the same function.

## Inversions

The same comparison ranks one way in one layer and the other way in another.

| Of two forms, the one that ... | ranks higher in | ranks lower in |
|---|---|---|
| keeps serving through another party's failure | M (M4) | C and D: a call that outlives its failure (C6) |
| is specialized to this model and these machines | D, instances at load (D11, D2) | C (C2) and M (M1): the transport serves every model alike |
| has members knowing more of each other's parts | C (C7) | M (M2): bridges agree only on a pairing; D (D7): topology is unknown |
| has members doing different work | D (D5) | C (C7): a member deciding alone desynchronizes the schedule |
| adds a feature that measures faster | D (D10) | M (M1): every feature is state with the node's lifetime |
| lives longer | M: resident, killable | D: what outlives a call, other than caches keyed by content, ranks lower (D8) |
| has data kernels write the stream's ring slots and ready words themselves | D (D10): no copy between them and the wire | M (M5): clients hold the addresses of the bridge's words |

Of the last row, the operator, 10-20: "if this is the fastest possible operation that the hardware allows, we might need
to allow it; memory safety is not a requirement but zero reducible overhead is a requirement".

In every layer, a form that adds a mechanism to survive another layer's failure ranks below the form that removes the
failure at its owner (M4). The operator, 10-18: "features which call for error correction or protocols are almost
without exception superfluous and wrapping or preserving another errorful feature".

## Two channels: the slow reliable one and the fast prenegotiated one

The operator, 10-21, verbatim: "when would tcp notionally be allowed if you have tow connection modes: one that is fast
and must be prenegotiated so that it can happen almost instantly, and another that is reliable but hundreds of times
slower? the answer will not be all or nothing." And of the answer below: "All of these are correct. ... from this we will
realize that the pairwise ranking rubric exists for a reason: to tease apart antinomies that claudes will try to invent
when tehy are not sure what to do wrt multiple agendas that are active at once and demand different hings at different
times" (10-21). Not "no ssh", and not a slow handshake between every layer: the slow channel ranks by where its cost
lands relative to the fast path it serves.

Where the slow channel (TCP, ssh) ranks higher: wherever the fast path cannot exist yet, paid once per long-lived object.
- Pairing a cable: before the queue pairs are made and connected, RDMA has no channel at all; the bridge exchanges
  queue-pair numbers and GIDs over its control socket once a link (M2's "pairing of a new cable").
- A communicator's bootstrap: NCCL's own rendezvous over TCP sockets (C1, the literature's form; the operator kept its
  gather and first-position agreement, 10-19: "Keep both"); with communicators kept across calls (M8) its milliseconds
  are spread over thousands of calls, a few nanoseconds each.
- Administering the substrate: installing or upgrading a bridge, starting daemons, ssh as PRRTE uses it (10-16: "'ssh
  stays for one job: administering the substrate itself (installing or upgrading a bridge), as PRRTE uses it only to
  start its daemons.' sounds fine"); a cabled node reached over its cable and used (10-20).
- Rare records between bridges: the map's flooding, at the bridge's own time scale, never per transfer.

Where it ranks lower:
- On a per-call or per-step path: at 10⁴–10⁵ ns against about 2 × 10³ ns for a SEND into a posted receive (crossings
  measured at 1.9 µs, 10-17), a TCP exchange per call is a few percent of a 3 ms step against "1% overhead is not
  acceptable" (M8); per step, worse. ssh per call is 10⁶–10⁸ ns: "are you trying to add ssh or a command plane for no
  reason even though this sounds like it introduces preventable latency? ssh isn't a reliable connection so this sounsd
  like a way to wriggle a bespoke protocol back into the code here yet again... i better not hear anything about 'you
  start code by hand' either since that is clearly not how megatron-lm starts up in pytorch" (10-19).
- Deciding the fast path's state while it runs: a TCP END closing an RDMA transfer (M2: "the closer's END on the TCP
  control socket ≺ the completion on the transfer's own queue pair"), a TCP-timed detector concluding an RDMA link: "why
  is there a separate network interface using a separate network protocol wtih different latency timing than the rdma
  underpinning the transport doing random things to bridges???" (10-20). The slow channel has its own failure and timing
  domain, and does not govern a channel whose failures are its own completions.
- Carrying data or calls the fast path could carry: a call's text on the host path, ssh to start a call's part. And
  libraries, transport or compute blocking on ssh ≺ none; a test script using ssh is not ranked: "aight if libs and hteu
  nderlying transport and compute are never blocking on ssh it's fine, it's not a seriosu problem if a test script uses
  ssh (not *really*)" (10-07).

So the slow channel bootstraps and administers the fast path's long-lived objects; it is never on the per-call path, and
it never decides a running RDMA channel's state. A call between machines is data landing in a resident program's posted
buffers (M1: "a client which runs very fast when the right buffers have just appeared asynchronously in memory").

## Failure across interfaces

The operator, 10-20, verbatim:
- "to me, the only possible reason for your weird statefulness reflects a sickly and perverted desire to 'prevent data
  processing code from ever failing explicitly' by 'treating failure of collective ops and bridges as unexceptional,
  typical, common, or desirable'. this is not correct. review the rubric and clarify positions on whether failures are
  contagious or allowed to explode across interfaces, and whether there are any guarantees of lossless operation for
  data processing functions (hint: no. there are no guarantees for lossless operation to data processing functions.
  however, empirical loss must be very low from the nested requirements that collective operations almost never fail
  and bridge operations almost never fail.)"
- "it is not possible for the data processing code to succeed while all lower substrates have failed. therefore the
  desired tradeoffs that seem to eb implied in past work are not tradeoffs but pure regressions or fault injections into
  the source, which is why they user is authoritative on removal without replacement."

The positions, in every kind, as ranked pairs:
1. **Failure is exceptional.** Code that treats a bridge's or a collective's failure as ordinary ≺ code that treats it
   as exceptional: bridge and collective operations almost never fail (Quantities: errors), and data processing's
   empirical loss is very low because those two are nested beneath it, and for no other reason.
2. **No data processing function is guaranteed lossless.** A data function made to succeed while its substrate has
   failed (waiting out a peer, retrying, refusing, reconstructing, holding state across the failure, reloading what the
   failure took) ≺ one that fails and returns its error through its error channel, a value beside its result (a tuple's
   second index), for its caller to run again on the map as it is then (C6); the former is a regression, a fault
   injected into the source, and ranks below its removal without replacement.
3. **Failures are not contagious.** A failure that reaches past its own caller (the bridge failing a client for another
   client's failure, a collective's failure failing more than that call, a data function's more than its call; an
   interface carrying a refusal, a stuck state, or a condition one party must clear for another) ≺ a failure that
   belongs to the party that failed and reaches only its own caller, as a value (M4, C6, 10-20).
4. **Transactions fail; a healthy node never does.** "transactions can fail, nodes that are healthy can never fail. that
   means that a bridge can fail then restart and reconnect, that a collective operation can fail and repeer, that a data
   processing operation can fail and relaunch. inducing a system halt or kernel halt causes the entire node to fail.
   that is not permittable for obvious reasons." (10-20). A halt of the system or the kernel, at any layer ≺ a failed
   transaction: the Mini's watchdog panic of 10-20 (a rank running on a stopped bridge's window beside the new one,
   wiring all its memory) is the case.
5. **Rare failures get no machinery.** A protocol, refusal, stuck state, or handler that changes data code from outside
   a call, built to make a rare failure ordinary ≺ none; something set up per call to survive a rare event ≺ whatever
   would be faster kept, kept (M8, 10-20); a bridge restarted thousands of times a second, minute, hour or day ≺ one
   that is not.
6. **Every program can be shut down whole, and a client transduces its lower layer's end as a value.** The operator,
   10-20, verbatim:
   - "we need to be able to service and maintain these machines, incl. to deliberately free memory because a bridge is
     holding too much for reasons we do not anticipate."
   - "1: do not stop meshes hundreds of times per second, minute, or hour 2: allow programs to be completely shut down
     and release all resources they are reserving from lower level kernels, e.g. os level kernels or firmware for i/o
     devices 3: 3 nines of reliability is a higher standard than infallible, as attempts to write infallible code have
     produced programs with 10% uptime or lower."
   - "clients have to transduce errors to the data processing code using them for collectives. it's up to you to decide
     how this works procedurally, but it isn't going to be by stack manipulation or an exception which tries to violate
     or interrupt code in a different interface, rather than by a return-with-a-value, an explicit and different error
     channel instead of a sentinel value, and the client either closing if it cannot accept a new memory allocation from
     a new bridge or continuing to exist without access to the allocation or memory mapping object it used that referred
     to the old expired bridge."
   - On launchd's relaunch of the bridge racing a release: "launchd launches mesh_heartbeat. mesh_heartbeat is a service
     which is on a different thread than mesh. mesh_heartbeat uses polite opens and polite closes on the actual mesh
     service."

   So: a release stops the bridge (its agents' exits free every device object) and expires its window; no party signals
   a client for it; each client's next lease or communicator returns the expiry as a value (ncclRemoteError, and
   ncclMeshWindowExpired), and the client frees what it holds of that window or ends; the window is unlinked, and its
   memory the system's, once none of its holders lives. One supervisor (the heartbeat) opens and closes the bridge on kernel
   events (a delay only bounds how soon a bridge that ended unasked opens again), and no second window is made beside a holder of the first. Ranked: a deliberate stop
   that leaves memory held until reboot ≺ a release that frees it; launchd relaunching a stopped bridge ≺ a heartbeat
   that closes it and keeps it closed; a bridge ending its clients for a window change ≺ clients that receive the expiry
   as a value and let go; a restart policy of one second ≺ one bridge start a minute at most (REOPEN_S).

## Strings are messages

The operator, 10-20, verbatim:
- "consider that an error which is not an exception or kernel fault within the calling context is literally nothing but
  a message to transduce along a 'rail' of functions which return a value and either an empty message or a nonempty
  message, and that code should not sympathetically shit its pants because a string happens to have `error` in utf8
  somewehre in nromal operation."
- "functions with operands of type string ought not produce side effects. functions with operands of type string ought
  not return anything but a string. does this mean that we don't write code like 'lenstring(*)->int'? yes. we do not
  write code like that and do not use it anywhere inside of the three scopes of concern: bridge, client, dataprocessor.
  other unrelated programs which do tabulation, visualization, bouncy gummy three dimensional solids which tilt and
  jiggle with the direction and size of rdma wire crossings are unaffected as they are not identifeid by the three
  programkinds in the rubric."
- "to make this even more pair rankable in the rubric's terms: lenstring(*)->string #expect ascii printed string of
  number, e.g. `four`, `fourhundred`, `fourtythree`. a function with the type just described would be allowed, as it
  would not be parseable as an integer by badly behaved or dispositionally evil kernel functions; we can imagine a
  logger which writes numbers as strings of chars to avoid even the superficial apperance of possible misconduct."

The positions rank; they are not rules of total exclusion. The operator, 10-20, verbatim: "we might have a narrow
exception to the topic we are discussing if we are talking about 'how we will load a safetensors file given that
safetensors use graphs of strings as metadata (fine actually idc)'; this is why the rubric keeps talking about
rankability rather than rules of total exclusion or inclusion". A format whose metadata is text (a safetensors header's
tensor table, a model's config.json, GGUF keys) is read at its load, once, into typed values (offsets, shapes, dtypes,
the declared structure); that ranks above text read again on a recurring path, and above structure inferred from names
where a declaration exists.

The operator, 10-20, verbatim, on logging and on names an operating system's interface takes: "the rubric is about
pairwise relationships, not absolutes. add to the rubric that logging is side effectful and takes strings as operands
yet nevertheless cannot be reduced away without compromising on other values, while removing many otehr string typed
functions is possible without compromising on our values. integrity is important here, and is emphasized by the rubric
based approach rather htan 'rule' based approaches." On how the export finds which operator belongs to which layer and
role: "either of these in a module is acceptable [a declaration we write ourselves, or the partition derived from
config.json], as is using the stock litert engine without modification as it is not code we are implementing a priori;
this is similar to the claims about rankability wrt the rubric for the other otpic".

So each string site is ranked against what removing it would cost, not struck by a rule:
- **Logging** takes strings and has a side effect, and still ranks above its absence: a log is how a person sees what a
  node did, and removing it gives up another value (servicing a node, rubrics 6) for nothing. What ranks lower is a
  log's text read back as a value (the driver parsing a rank's lines), or a log written on a recurring path where a
  counter would do.
- **A name an interface we do not implement takes** (the operating system's `shm_open`, a spawned image's path, a
  device by name, the stock LiteRT engine used without modification) is that interface's form: the code we write
  around it is ranked, not the interface.
- **Most other string-typed functions remove without compromising any value** (a record kept as a string-keyed
  dictionary, an op chosen by comparing its text, a kernel looked up by name on every dispatch, a structure inferred
  from names that a declaration states), and rank below their typed forms.
- **The export's partition** (which operator belongs to which layer and role): read from tensor names ≺ a module's
  declaration we write, or one derived from config.json; or the stock LiteRT engine run unmodified.

The positions, in M, C and D, as ranked pairs:
1. **A string is a message.** A function that takes a string and returns other than a string, or has a side effect
   (measuring, comparing, searching, parsing, hashing, looking up by, or converting a string into a value; a branch on a
   message's content) ≺ a string carried, composed (string to string), and returned beside a value; with the weighing
   above (logging, an interface's names).
2. **Every function's result is a value and a message**, the message empty or not. Code that changes what it does
   because of what a message says ≺ code that carries it: a fault is an exception or a kernel fault in the calling
   context, and anything else is a message on the rail.
3. **What crosses into these programs as a value**: configuration, requests, identities or outcomes as text the
   program parses ≺ as typed values.
4. **Programs outside the three kinds** (tabulation, visualization, renderers of the wire's crossings) are not ranked
   here: they read and write text as they like.
5. **Types and structure are declared, not inferred at run time.** The operator, 10-20, verbatim: "how is a
   model.config.file written in the huggingface et al dialect? i do not think it is by type inferenec at runtime.
   figure out why the user is being picky about type inferences and then act on it". A model's configuration declares
   its structure as typed fields (gemma-4-E2B's `num_hidden_layers` 35, `layer_types`, `num_attention_heads`,
   `num_key_value_heads`, `head_dim`, `intermediate_size`, `num_kv_shared_layers`, `enable_moe_block`,
   `attention_k_eq_v`), read once into a typed object that the code reads by field. Inferring the same facts while the
   program runs (a layer's index from a regex over a tensor's name, MoE from whether a tensor of some name exists, a
   value's type from a cast that may fail, a format from a file's suffix) is a guess: it costs work at every load or
   call where a declared value costs one read, and when the names or the shapes change it yields a different program
   rather than an error (the operator, 10-19: no parameter derived or guessed inside a function). Ranked: inferred ≺
   declared and read once, at the boundary, into a typed value every consumer takes.
6. **What a call returns**: text formatted by one program and parsed by another ≺ a value a consumer matches exactly,
   a word or bit vector compared against the expected one. The operator, 10-20: "i do not understand why serializing or
   deserializing or checking strings is the exact same expected overhead as a bitvector exact match in a kernel which
   takes a return and does something in response...?"

Ranked by where the text is read, lowest first: text read or compared on a recurring path (per frame, dispatch,
window, step or call) ≺ text read once per load or start into typed values (a safetensors header, a declaration) ≺ no
text.

Ranked, in every kind: `lenstring(*)->int` ≺ `lenstring(*)->string`, its number spelled in words (`fourtythree`), which
nothing can parse back into a value; structure inferred from names, presence or casts while the program runs ≺ structure declared and read
once into a typed value; a call's outcome as text the driver parses (`'committed'`, `{"failed": {"error": …}}`) ≺ a
word each member sets that the driver matches exactly; a message whose numbers are digits another program reads back as values ≺ one
whose numbers no program can read back (the operator's imagined logger); a rank reply's `"error"` key moving an
allocation ≺ a typed outcome value; a request parsed from
JSON text inside the rank ≺ a typed request; `strcmp`, `strlen`, `atoi` or a name's hash in the bridge or the library ≺
a numeric constant or a typed operand; logging on a change found by comparing strings ≺ no comparison.

## How the operator reviews

- **Rankable, never pass/fail.** "rankable choices are broadly good; ... pass/fail tests are generally not good; ...
  writing a specification in two places, once in source and once in a test is duplication of code" (10-14).
- **A rubric, not rules.** "a rubric for analyzing static code ... multiple 'sorta orthogonal' concerns which are each
  clearly scored separately ... some very abstract and high level concerns, and a few very limited in scope fine
  grained mesh/bridge specific concerns" (10-19).
- **Reviews end in source.** "A writes source, B writes annotation, A&B return source (not a refusal to write source)"
  (10-14).
- **Few transformations.** "it's far more likely there's 4 things to do than 32" (10-14).
- **Data flow, not stated reasons.** "get rid of stated reasons and rules ... none of them seem to present a data flow
  level view of the code" (10-19).
- **No design choices.** "there is a single and particular hardware implementation available. it is poorly abstracted,
  wrapped, or scheduled by other code. therefore we have to abstract, wrap, and schedule it" (10-16).
- **Fix what you find.** "fix problems when you find them instead of remarking on them" (10-19).
- **Reviewed before built.** "the thing called a rubric is not metaphorical. all code needs to be reviewed in terms of
  the rubric before being built and compiled from this point onwards" (10-20): a change's ranked pairs, dimension by
  dimension, are written before it is compiled; "use a subagent to review code which has been newly written within a
  session wtih the rubric file" (10-20).
- **The one rule is the review.** "this is not a 'rule'; if there are 'rules' they are about the use of the rubric to
  review code with independent subagents to mitigate and prevent complicated, labrythine, slow interventions and their
  removal" (10-20). Everything else here ranks.
- **The rubric answers the choice.** "why are you asking me to make a choice when the rubricfile exists to totally
  answer questions of this kind?" (10-20). Ranked: asking the operator to choose between forms the rubric ranks ≺
  choosing by the ranking.
- **Write the module, not the history.** "identify hwat the code did and write that module then, rahter htan playing
  pretend wrt git version trees etc" (10-20). Ranked, where a rejected change is entangled with later work: recovering
  the earlier code from version history ≺ reading what it did from its source and writing that module now.
- **Megatron TP is its semantics.** "megatron-tp meant something very semantically sepecific and it can be recovered
  from source alone" (10-20): every rank runs the same program from the same inputs (metal-microbench
  tools/torch_parallel/tp.py, 10-02: wq, wk, wv, w1, w3 column-parallel, wo, w2 row-parallel, an all-reduce after
  each), started as torchrun starts it; only tensors cross between ranks, in the program's collectives; Megatron-LM's
  inference server has rank 0 broadcast a fixed-size typed value and the token tensor, the others looping on that
  broadcast. Ranked: members as servers interpreting a head's request (d673284, 10-19) ≺ every rank running the same
  program from the same inputs (10-20).
- **No part ends another.** "i do not want to hear anything about 'when x died, y co-died ...' ever again" (10-20).
  Ranked: a coupling through which one part's end ends another ≺ none; such a coupling told as the account of a
  failure ≺ the coupling found in the code and removed.

## Where HEAD sits (2026-10-20), as ranked pairs

- **M2, M3, M4, M5:** v127's bridge (a binding protocol in client slots, links re-paired inside the process by RESET and
  ERR, clients claiming lane pairs) ≺ HEAD's (version 128, 0e34c13e; the M5's bridge since 10-20 05:53 UTC): a stream a
  link held by an owner word, queue pairs made and connected once, nothing a client asks of it.
- **M3:** the deleted kill guard ≺ HEAD after its deletion.
- **C1:** libnccl-mesh exporting its own Metal functions and including the planner in nccl.h ≺ HEAD's ABI of NCCL's
  functions, their twins and its extensions (5312f9b2).
- **C2, C7:** one communicator a node pair (version 128 with 287795f5: a second toward a node refused, EBUSY, as
  metal-microbench's rank process makes beside its standing group) ≺ HEAD's four streams a link (version 129), each
  communicator holding its own toward each member's node, taken by the identity free at both ends, as v127's four lane
  pairs a link allowed without its binding.
- **C6, M8 (10-20):** communicators made again every call (metal-microbench 906726c, 21–34 ms a call) ≺ communicators
  kept for as long as they work, a failed one reporting its failure through `ncclCommGetAsyncError` and made again
  between calls: "1% overhead is not acceptable" (10-20). C6's lower pole is state the next call must repair, not
  persistence.
- **D7:** goal_a.swift reading its bridge's records to find its node ≺ the host's own name (metal-microbench 5e95b12);
  mesh_rank.m asking libnccl-mesh for its node's fabric address ≺ the configured fabric address that binds here
  (configs/driver.json and bind(2)); rank.swift polling `ncclMeshMemLive` each second, and timing init waits by
  `ncclMeshInitWaits` ≺ a failed call that finds its leases stale through NCCL's `ncclMemAlloc` (the operator's choice,
  10-20).
- **D2, D3:** attention plans and instances derived in builders (before 2b75b6a and d3d0f4b) ≺ synthesized at load.
- **C4, C7:** a group's completion as one serial order of dispatches that every rank had to build alike (order_group:
  a wait cycle among three or more ranks) ≺ each channel progressing on its own (264ce849, the operator's choice of
  10-20), the two computing one function on the pair (KL between their records 6.2 × 10⁻⁷).
- **M4:** a dud's all-ones bytes read by its consumer as a failure of its own group ≺ a dud landed as bytes (fcb79df2,
  the operator 10-20: "'duds' are not sentinel values").
- **M4, M6:** a link lost after it connected staying down until a person relaunched the bridge (version 129) ≺ a
  process a port (version 130, 1f9be1d5): the lost link's agent ends once its quiet window has passed, the next agent
  pairs it again with nobody acting, and calls in flight fail as values: on the pair, 10-20, the M5 paired again 7.4 s
  after the Mini's bridge stopped.
- **M3, M4:** a stream kept for a lost pairing's holder, bootstrap waits with no bound, and a member's stale leases
  refusing its group until a call the refusal blocks (version 130) ≺ version 131 (9add4cbe, ab2e5192, metal-microbench
  7b04f33): every stream new at a pairing, every bootstrap wait a slice within the platform deadline, and the refused
  member retiring its residents at the refusal. A process holding leases of two regions (9add4cbe's P3) ≺ one region a
  process: the former wired all of the Mini's memory and its watchdog panicked it (10-20 18:27 UTC). On the pair after
  ab2e5192, the Mini's bridge restart failed one call and the next two committed, its wired memory steady.
- **M3, M4, rubrics 6 (10-20):** a window unlinked and made again at every bridge start, its clients' leases left in a
  dead window beside the new one (through version 131) ≺ a window adopted across a restart (923d73fd, layout 132: on
  the pair the Mini's bridge restarted and the next call committed, wired memory 20.0 GB throughout) ≺ that, with a
  release that frees it: a heartbeat opens and closes the bridge on kernel events, a release expires the window, each
  client receives the expiry as a value and lets go or ends, and the heartbeat unlinks the window once none of its
  holders (bridge, agents, clients, recorded by pid and start) lives, before any bridge makes another; a bridge that
  ended unasked opens again a minute later.
