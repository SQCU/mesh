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
- It ends when a signal arrives, and its exit releases everything.
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
5. Never write a rule, a memory note or a pass/fail check in place of a ranking.

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

**Quantities.** These are read alongside the rankings. They are indexed absolutely, so they are not part of any
ranking.
- Throughput in tok/s, per stream and aggregate.
- KL at the unembedding against LiteRT's pair KL. Up to 2.0 is allowed while shapes work is unfinished.
- A difference under 3% counts as equal.
- Latency more than 3 times the hardware's per hop is forbidden.
- No DMA or network channel goes 5 s without something happening on it.
- At least 1,000 normal frames per transport error.
- Testing takes about 12 s an hour; an investigation runs at most 10 s of intensive work.
- No collective has fewer than 2 members.
- After a backend's first build, switching to it costs under 1 s.

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
    loop and a watchdog OS restart" (10-20).

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
  - a node shutting down for an unknown reason during DMA ≺ a node that stays up (10-15).
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
    are transducing *is* a complete protocol. it's done. we're not writing a protocol." (10-20).

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
  - asking for administrative actions ≺ "services always respect shutdown isgnals no matter what state" (10-18).

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
  - host groups that poll for 1 ms and then park ≺ no parking (10-19).

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

**C6. Failure's reach.**
- Ranks higher: the form in which a failure ends less (the call alone) and leaves less for the next call.
- Lower pole: resumption inside a call, retries, waiting out a peer, and communicators kept across calls with state the
  next call must repair.
- Upper pole: the call returns an error, and the caller runs it again on the map as it is then.
- The operator's pair: lower, "overwrought bullshit to pretend *within-call* state is recoverable, *synchronically*";
  upper, "recover from any error *across* calls, *diachronically*" (10-01, docs/principles.md section 4).
- Ranked: within-call resumption ≺ "return an error (which a caller can presumably retry against immediately)" (10-14).

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
- Also: "bit identicality of floats is not real" (10-04).

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
- Lower pole: reading the bridge's region, watching its process, probing links per call, ssh or Wi-Fi to a cabled peer.
- Upper pole: topology unknown to the program; a node asks its own bridge what neighbours it has.
- Ranked:
  - "sshing into a terminal mode on a mesh peer" ≺ "querying their own mesh bridge" (10-16);
  - "flaky ssh" in a call's path ≺ none (10-07).

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
- **C6:** communicators kept across calls (metal-microbench mesh_rank.m `kept_sets`) ≺ calls that fail and are rerun.
- **D7:** goal_a.swift reading its bridge's records to find its node ≺ the host's own name (metal-microbench 5e95b12);
  mesh_rank.m asking libnccl-mesh for its node's fabric address ≺ the configured fabric address that binds here
  (configs/driver.json and bind(2)); rank.swift polling `ncclMeshMemLive` each second, and timing init waits by
  `ncclMeshInitWaits` ≺ a failed call that finds its leases stale through NCCL's `ncclMemAlloc` (the operator's choice,
  10-20).
- **D2, D3:** attention plans and instances derived in builders (before 2b75b6a and d3d0f4b) ≺ synthesized at load.
