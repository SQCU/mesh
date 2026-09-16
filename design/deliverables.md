# Deliverables — the completion matrix for the streaming partial-tensor collective library

Operator, 2026-09-15, verbatim:

> lets bundle up this big list of deliverables into a document which can be treated as a
> goal by either codexes, claudes, or claudes launching a `codex` as a foreign-api
> subagent to steer and give assignments; this list should be comprehensive enough that
> it's hard to confuse partial progress for total progress or invariants in design from
> omissions that are free to fill in with specailizations

> if the library was 15k lines, and it couldn't run a tp streaming collective operation
> with no overhead, almost without question all 15k of the lines gotta go. this can be
> derived by the number of expressible programs that can be written with 15k lines of
> code, and how little had been done on any of the serious deliverables. bad code is big
> and does nothing. bad code is bad because revising it is more complicated than
> deleting it. doing nothing while staying big takes proactive work.

> rdma links on thunderbolt are chuddy. add more requirements about recovering after
> network segmentation of the form seen in cable disconnects / reconnects (two peers
> exist, but where the thunderbolt bridge goes on each side is a new logical/indexed
> connection between two peers who can very quickly reestablish who they are and what
> they'll be doing. recovery only has to happen between NFEs; there is no error recovery
> guarantee within a NFE.)

This document is the goal. It is handed verbatim to a Codex session, a Claude session,
or a Claude that launches `codex` as a subagent. Every row is a deliverable with a
signature, a reference to cross-implement from, and a check readable from source; the
work is finished when every row is ✓; a row that is not listed is not work; an
invariant (§1) is never a row an agent may fill in with a specialization. The
operator's other normative sentences are collected verbatim in
[collective-goals.md](collective-goals.md) and [SPECIFICATION.md](SPECIFICATION.md) §25.

Operator clarification, 2026-09-15, superseding the former I17/I18 and X3 wording:

> i18 was misspecification and described guards or blocks which, ironically, keep
> showing up in library code as control flow that prevents values from being
> drained or transacted.

> why is there a data structure called a 'page table' and how does this allow
> solutions to the problem of data arriving in interleaved orders, e.g. if there's
> a ring topology where every node has at least 2 peers? do we even care if some
> rows appear in the 'wrong' order then need to be shuffled in placement to become
> fully contiguous later for contiguous-typed operations? (no, this can be done
> asynchronously and very quickly...).

Dedicated queue progress and indexed page placement are required mechanisms.
They do not authorize an operation wait, a reuse guard, or withholding an
independent value. Logical coordinates are realized ahead of execution; their
backing need not have an immutable physical address. This correction preserves
the deliverables and their unfinished work; it supplies no performance evidence.

## 0. How to use this document as a goal

1. Read §5. Take the **first row whose status is ✗ or ◐**, in order, unless a steering
   agent assigns a specific row.
2. The assignment is exactly that row's *Signature*. Produce it in the named file(s),
   using the named reference; nothing else changes.
3. Run the row's *Check*. If it holds, set the row to ✓ with the commit hash, commit
   with the row id in the subject, **push**, stop.
4. If the row cannot be done without a symbol that does not exist, do not invent it:
   add the missing row to §5 as a new ✗ line *above* the current one, with a signature
   and a check, and stop. That is progress; a wrapper is not.
5. Anything in §6 is reverted on sight, regardless of who committed it.

"Continue working toward the goal" means "take the next row", never "improve the code".

## 1. Invariants (never rows; violations are defects wherever they appear)

| id | Invariant | Check (source, not runs) |
|---|---|---|
| I1 | Two-sided only. `IBV_WR_SEND` into a posted RECV; never WRITE/READ/atomics (TN3205; hellas `-ENOTSUPP`). | `grep -n "IBV_WR_RDMA\|rkey\|remote_addr" rdma/` empty |
| I2 | Producers never wait. Publication happens when the writes are visible and returns. | no `wait`/`semaphore`/`poll` on the path from numerical completion to `mesh_publish` |
| I3 | Presence = receive completion. No acknowledgement, handshake, health check, credit protocol, or in-band sentinel in the data path. | `grep -n "ack\|handshake\|ready byte\|sentinel" rdma/mesh-flow.c rdma/mesh-call.c` empty outside the one-time out-of-band pairing |
| I4 | Memory mapping, not control flow. Every simultaneously live value has its own registered pages; no reuse guard, claim check, occupancy check or slot-modulo. | `grep -n "claim\|EBUSY\|occup\|modulo" rdma/mesh-call.c rdma/mesh-dataflow.c` empty |
| I5 | Lifetime is reference counting owned by mesh; callers never free. Release returns pages to the pool without zeroing. | no public value `free`/`release` in `swift/Mesh.swift`; `mesh_collect` has no `memset` |
| I6 | RDMA TX and RX each own a hardware thread per link and share it with nothing. All queues drain and refill immediately. | one thread per link per direction in `rdma/mesh-flow.c`; no numerical call on those threads |
| I7 | Mesh contains no numerics. Matmul, activation, norm, sum, max, attention, sampling are supplied `TensorFunction`s. | `grep -niE "matmul|gelu|softmax|rmsnorm|vDSP|MPSMatrix" swift/Mesh.swift rdma/*.c` empty |
| I8 | Mesh contains no application. No model, layer, game, policy, or caller name in the library. | `grep -rniE "gemma|xonotic|havocbot|layer|residual|policy" swift/Mesh.swift rdma/` empty |
| I9 | Caller placement only. No cost model, automatic sharding, topology inference, or collective inferred from rank count or tensor type. | `grep -niE "cost|estimate|choose|infer|auto" swift/Mesh.swift` empty; `scatter`/`gather` keep explicit destinations at every world size |
| I10 | Transport fragmentation is invisible. A section's bytes are not bounded by one request; chunking changes no shape, partition, function, or verb. | no public capacity query; `mesh_section_create` takes bytes |
| I11 | Explicit synchronization exists only as a diegetic library call (`syncOnRemoteFill`) and is never a default. | the only non-example caller of `mesh_sync_on_remote_fill` is `Mesh.syncOnRemoteFill` |
| I12 | Failures are `Result`/status values out of band; an operation concludes early on link error; the driver re-invokes at the data layer. | no `exit`/`abort` on a verbs error; `mesh_call_fail` records, never blocks |
| I13 | Amdahl superiority is the only legitimacy, measured on the public path. Compilation, deletion, documentation, and examples are not evidence. | every performance claim cites a run through the server endpoint with solo and n-node numbers and Karp–Flatt e |
| I14 | Every public construct cites one published mechanism, once, in `algorithm-sources.md`. Bibliography growth is not progress. | one section per symbol in §2; no per-commit sections |
| I15 | No tests, gold harnesses, evidence JSON, provenance records, or trace exports. Verification is source reading plus §5 checks plus public measurements. | `measurements/` gains only public-endpoint runs |
| I16 | Size is evidence. A library larger than the §5 rows need, while any of rows 13–19 is ✗, is deleted and rewritten from §2–§3, not revised: revision cost scales with what exists, deletion cost does not. Reference size class: MLX distributed (8 functions), JACCL (~1.5k lines). | `wc -l swift/Mesh.swift rdma/*.c rdma/*.h` ≤ 3,000 while rows 13–19 are open; a commit that grows the library without flipping a row is reverted |
| I17 | Indexed dataflow on the hot path. Realize logical rows, consumer ranges, storage capacity and backend bindings before execution. Runtime uses compact arrays for page mappings, queues, counts and operands; arrival-derived page indices and asynchronous placement are permitted. Remove linked-list traversal, avoidable metadata reconstruction and software transport admission gates. Physical arrival order never defines tensor coordinates or collective semantics. | trace each received chunk through its peer-qualified logical row, canonical backing and supplied operand; independent arrival orders preserve values without imposing a send order |
| I18 | No guards or blocks that withhold ready work. Producers publish on visibility and return; dedicated TX/RX threads continuously drain completions and post available work. Queue polling, queue-empty observations, indexed placement, native status handling and completion-driven dependency/refcount updates are permitted. No implicit barrier, acknowledgement, reuse guard, blocking retry or wait for a particular value may hold up independent publication, transfer or consumption. Layout conversion is asynchronous dataflow for its own operands. | group W audits actual withholding and serialization, not the spelling of a loop, conditional or index operation; I11 remains the sole explicit operation-sync path |
| I19 | Recovery happens between NFEs, never inside one. A link failure concludes the in-flight instances as `Result` values (N2) within one completion; no path in TX/RX/publish/invoke retries, re-sends, or waits for a link to return. Re-pairing is the bridge's own bounded loop; resumption is the driver's next `submit`. The bridge never exits and never wedges the provider on a lost link (`RDMA-RULES.md`: destroy QP/CQ, deregister, re-create; never leave a verbs call blocked). | `grep -n "retry\|resend\|reconnect" rdma/mesh-flow.c rdma/mesh-call.c` hits only the bridge's link controller, never a data-path function; group R rows |
| I20 | Rebuilding or launching the bridge never raises a new networking permission. Every executable that accepts connections is one binary at one install path per node (`/usr/local/mesh/bin/mesh-flow`), signed with a persistent identity (`Mesh Bridge`) and a fixed identifier (`io.mesh.bridge`), allow-listed once; rebuilds replace it in place; the launchd plist never points into a worktree, `.build`, or scratch path; clients never `listen()`. | `codesign -dv /usr/local/mesh/bin/mesh-flow` shows `Identifier=io.mesh.bridge` and a non-adhoc signature on both nodes; `bin/mesh-bridge.sh` refuses worktree/build paths; `socketfilterfw --listapps` has exactly one mesh entry per node; `grep -n "listen(" swift rdma/mesh-call.c rdma/mesh-dataflow.c` empty |

## 2. Public surface (closed)

The importable library is the Swift module `Mesh` (`swift/Mesh.swift`) over the C
runtime (`rdma/mesh-call.c`, `mesh-dataflow.c`, `mesh-flow.c`, `mesh-verbs.h`, `mesh.h`):

```
Mesh(region:, rank:, size:, workers:, count:, placement:) // N1 changes `count` to in-flight instances
Placement(owners:, routes:, work:, bytes:, cuts:, path:)  // explicit ownership, directed routes, optional bound inputs
Placement.Edge(source, destination)                     // ordered endpoints; route excludes source and includes destination
Mesh.tensor(on:sections:) -> [TensorPart]                 // partial tensor = contiguous sections, each with a rank
TensorPart.partial: Bool                                 // contribution view pending reduction
MeshError.partialOperand(TensorPart)                     // invalid use reported during setup
Mesh.constant(on:bytes:initialize:) -> TensorPart
TensorFunction.cpu / .metal(device) / .prediction(model) // supplied numerics; input/output view factories
Mesh.call(function, inputs:, outputs:, on:, worker:)
Mesh.map(function, inputs: [[TensorPart]], outputs:, …)
Mesh.send(part, to:, queue:) -> TensorPart
Mesh.broadcast / scatter / allScatter / gather / allGather / allToAll
Mesh.reduce(parts, to:, using:) / reduceScatter(contributions, to:, using:) / allReduce(…)
Mesh.start()                                              // realize storage, bindings, receives, links
Mesh.submit(index)                                        // one value instance through the realized graph
Mesh.result(index) -> Result<Void, MeshError>              // N2: one status load; busy until conclusion
MeshError.busy / .link(peer:code:) / .function(call:code:)
Mesh.syncOnRemoteFill(parts, index:)                      // explicit, diegetic, never default
bounds(program, placement, topology) -> Bounds            // pure lower bounds; never picks a placement
```

Adding a symbol is a change to this section first (a ✗ row in §5), then code. A symbol
not listed is not part of the library and is deleted with its callers.

## 3. References to cross-implement from

| Construct | Reference | What to copy |
|---|---|---|
| transport | Apple TN3205; MLX JACCL `rdma.cpp` (`IBV_QPT_UC`, sge 1, `ibv_poll_cq` only) | QP/MR/CQ setup; posted-RECV credit; frame matching |
| progress engine | NCCL `proxy.cc`; MSCCL++ PortChannel proxy | one poller per link/direction that never computes |
| presence firing | Monsoon/TTDA; I-structures; Realm events | operand-associated presence; completion event fires dependents |
| partial tensor | Pallas `BlockSpec`; DaCe memlet; ScaLAPACK descriptor | sections with static bytes; placement as a value |
| collectives | Rabenseifner 2004; Patarasuk–Yuan 2009; Gloo `allreduce.cc`; JACCL `mesh_impl.h`/`ring_impl.h`; MPI-4.1 ch. 5 | RS+AG composition; direct vs ring; verbs' value relations |
| contraction streaming | SUMMA (van de Geijn & Watts 1997); Pallas collective matmul | K-panel rank-k updates published per tile |
| partial type | PyTorch DTensor `Partial`; Legion `reduce` privilege; Korthikanti 2022 | pending-reduction as a type; nonlinearity after full reduce |
| lifetime | RCU grace period; LMAX Disruptor; Lamport 1977 | reclaim after last reader; single-writer stamps |
| instances | MPI-4 persistent collectives; Pathways; CUDA graphs | realize once, submit per instance with distinct storage |
| topology/bounds | Patarasuk–Yuan (spanning tree suffices); Hockney; LogGP; COSMA | routes over any graph; cut/path lower bounds |
| legitimacy | Amdahl 1967; Karp–Flatt 1990; Pope et al. 2023 | measured serial fraction; TP decode/prefill bound |

## 4. Deliverables

Each: **Signature** · **Reference** · **Check** · **Not it**.

### S — substrate

**S1. Links are a configured list; a bridge serves them all.** `etc/bridge.conf` `links=('device,peer,local,remote[,service]' …)`; `mesh_peer_channel(ctx, peer, queue)`. Ref: JACCL hostfile. Check: no `peer=` scalar; `grep -n "size == 2\|rank == 1 - " swift rdma` empty. Not it: a hostfile assuming full mesh.

**S2. TX/RX per link on dedicated threads; drain and refill immediately.** `rdma/mesh-flow.c` `tx_thread(link)`, `rx_thread(link)`; RX reposts before delivering. Ref: NCCL proxy; JACCL per-wire polling. Check: I6. Not it: a progress loop that also runs numerical callbacks.

**S3. Chunked transport, invisible.** `mesh_section_create(ctx, bytes, count, receive, &section)`; internal `C`, `K = ceil(N/C)`. Ref: TN3205 frame matching; llama.cpp PR #26421 (128 KiB stride cost). Check: I10.

**S4. Out-of-band pairing once; versioned exchange; every syscall bounded.** `verbs_up(link)` with `XMAGIC+MESH_VERSION`; `accept/connect/read` each with a deadline. Ref: TN3205; `RDMA-RULES.md`. Check: each of those calls in `rdma/mesh-verbs.h` is followed by a timeout.

### L — library core

**L1. Partial tensor = sections with ranks.** `TensorPart(rank, bytes)`; `Mesh.tensor(on:sections:)`. Ref: Pallas BlockSpec. Check: a section is contiguous locally, addressable globally by `(rank, section)`; no hidden dense copy (I4).

**L2. Supplied function over sections.** `TensorFunction.cpu/.metal/.prediction`; `Mesh.call/map`. Ref: Pallas `pallas_call`; StarPU codelets. Check: I7; a CPU, a Metal and a Core ML function each bound through the same `call`. Not it: an expression compiler; a kernel zoo.

**L3. Publish on visibility, fire on presence.** completion → `mesh_publish(row)` → for each consumer in the row's contiguous use range: decrement its pending count; zero enqueues it on its worker. No consumer evaluates "are my inputs present". Ref: Monsoon; Realm; TensorFlow Send/Recv; Naiad occurrence counts. Check: I2, I3, I17; `mesh_publish` walks one contiguous range and performs one decrement per use.

**L4. Ten collective verbs as movement + supplied combine.** §2 list; `reduce*` take `using: TensorFunction`. Ref: MPI-4.1 ch. 5; Rabenseifner/Patarasuk–Yuan; Gloo. Check: every verb is `send` + `call(combine)`; movement does no arithmetic; explicit destinations at every world size. Not it: gather-everything-then-sum; any verb inferred from `size`.

**L5. Partial is a type.** `TensorPart.partial: Bool` set on `reduceScatter/reduce` contributions; `Mesh.call` throws `MeshError.partialOperand(part)` at setup when a non-combine function binds a partial. Ref: DTensor `Partial`; Legion `reduce`; Korthikanti 2022. Check: one `throw` in `call`; no runtime state.

**L6. Combine is any associative supplied function.** `reduce(…, using:)`; explicit binary tree. Ref: `MPI_Op`; Blelloch. Check: no special case for `+`.

**L7. Lifetime without explicit free.** internal `mesh_buffer_retain/release`; uses count down on numerical completion, TX completion, destruction; `mesh_collect` returns pages unzeroed. Ref: RCU; Disruptor. Check: I5.

**L8. Explicit sync as a library call with its counterexample.** `Mesh.syncOnRemoteFill`; `examples/sync-on-remote-fill.swift` `parallel|serial|deadlock`. Check: I11.

### G — genericity

**G1. Contraction is a two-operand `call`; K-partials are `reduceScatter` inputs.** caller: `call(dot, inputs:[x_k, w_k], outputs:[c_k])` per K-panel; `reduceScatter([[c_k]…], to:, using: add)`. Ref: SUMMA. Check: Gram `R(RᵀZ)` and FFN down-projection are the same calls with different operands; the Gram output feeds the next `call` with no copy. Not it: `nn.linear` inside mesh.

**G2. Indices as data.** `Mesh.map(function, inputs: [[x], [idx]], …)`; `idx` is a `TensorPart` whose presence gates the map; the supplied function does the indexed read. Ref: Pallas scalar prefetch; `jax.lax.gather`; Megablox. Check: routed-expert and neighbourhood callers exist outside mesh using only `map`; no routing code in mesh.

**G3. Two callers, zero specialization.** `metal-microbench` and `xonotic/solver` both `import Mesh`; mesh imports neither. Check: I8; both build against the same module.

### N — instances and lifecycle

**N1. Realize once, submit per instance, unbounded.** `Mesh(…, inFlight:)`; `submit(index) -> Result<Void, MeshError>`; the arena is sized at `start()` as inFlight × bytes-per-instance (SDF balance); an instance's pages return to a per-worker free list when its reference count reaches zero (an event, X5), and `submit` when no instance slot is free returns `MeshError.busy` immediately — never waits, never checks readers. Ref: MPI-4 persistent collectives; Pathways; CUDA graphs. Check: `submit(inFlight + k)` is legal and returns a value; `start()` prints bytes per in-flight instance; no allocation, no reader query, no wait inside `submit`.

**N2. Early conclusion as a value.** `Mesh.result(index) -> Result<Void, MeshError>` is one load of the instance's status word (no scan, no poll loop); `MeshError.link(peer, code)`, `.function(call, code)`, `.partialOperand(part)`. Ref: fail-stop; end-to-end argument. Check: I12; the Core ML chain concludes early on a killed peer and the driver re-realizes.

### T — topology and fleet

**T1. Topology is a graph value, observed.** `Topology(nodes:, links: [(a,b): [Link(device, bandwidth, latency)]])`; `Mesh.observe() -> Topology` from live pairings. Ref: JACCL hostfile as the full-mesh case. Check: tree, ring+spurs and full mesh run the same caller with only config changed; no code path assumes `links[(a,b)]` exists.

**T2. Routes belong to placement; mesh forwards on presence.** `Placement(owners:, routes: [(src,dst): [hop…]])`; `send` on a non-adjacent pair is one SEND per hop through canonical intermediate pages. Ref: Patarasuk–Yuan (spanning tree suffices); Pathways; Monsoon. Check: leaf→leaf on a star is two SENDs with no hub wait; per-cut bytes equal declared traffic.

**T3. Multi-link striping per pair.** `send(part, to:, queue:)` where `queue` selects among a pair's links; chunks round-robin when `queue == .all`. Ref: JACCL `ring_impl` wires; TN3205 10 QPs/device. Check: equal bytes per link of a pair in `report`.

**T4. Bounds before runs.** `bounds(program, placement, topology, capability) -> {compute, memory, cut: {cut: s}, path: s, max}`, pure; `topology` is the T1 value (links, observed), `capability: [node: Capability(rate, bandwidth)]` the caller's sustained rates. Ref: `docs/amdahl_superiority.md` four lower bounds; Hockney; LogGP; COSMA. Check: reproduces the ten-minute table (1+3, 2+17, 1+1); every report prints `measured / bounds.max`.

**T5. Retopology is a `Result`; the driver re-realizes.** `MeshError.topology(lost:, gained:)`; driver: `observe(); Mesh(topology, placement.restrict(topology)).start()`. Ref: fail-stop; Dean–Ghemawat; `RDMA-RULES.md`. Check: pull one cable mid-run on a ring: early conclusion names the link; re-realized program completes on the spanning tree; re-plug is visible to the next `observe()`.

**T6. Replicas are placement, not protocol.** `Placement(…, replicas: [section: [rank, rank]])`: a replicated section is produced on every listed rank as an ordinary extra output edge; each consumer is bound at `start()` to ONE replica (the placement's primary for that consumer); loss of the primary's link is a T5 retopology event whose re-realization binds the next replica. No read-time choice, no "first present" branch. Ref: Legion physical instances; end-to-end. Check: no runtime conditional selects among replicas; losing a link changes `bounds()` and triggers T5, not a branch.

**T7. Multi-tenant fabric by lease.** bridge serves N clients; `Mesh(…, lease: Lease(qpsPerLink:, arenaBytes:))`; setup returns errno when the lease exceeds `max_qp`/arena. Ref: NCCL communicator per job; PMIx. Check: two users' programs run on disjoint QPs/arena; `kill -TERM` of one leaves the other untouched; no `SIGKILL` path.

### W — waitless/guardless audit, one row per runtime path (shared by every other row)

Each W row traces the functions on one path and checks that ready work is drained,
published and transacted without an implicit operation wait or reuse guard. The
audit distinguishes queue progress, indexed placement and declared operand
dependencies from control flow that withholds independent work. Source and dataflow
are the evidence; counting conditional statements is not the check. Any later
commit touching a path re-audits its W row before its own row can flip.

**W1. Publish path** — numerical completion → `mesh_publish` → consumer range walk → enqueue. Files: `rdma/mesh-call.c` (`mesh_call_complete`, `mesh_publish`), `rdma/mesh-dataflow.c`.
**W2. Receive path** — CQ completion → presence store → publish. Files: `rdma/mesh-flow.c` (`rx_thread`, receive completion handler).
**W3. Send path** — publication notice → post chunks → completion → reference release. Files: `rdma/mesh-flow.c` (`tx_thread`, post loop, send completion).
**W4. Submit path** — `Mesh.submit` → `mesh_calls_submit` → root workers. Files: `swift/Mesh.swift`, `rdma/mesh-call.c`.
**W5. Result/collect path** — `Mesh.result`, refcount → free list, `mesh_collect`. Files: `rdma/mesh-call.c`, `rdma/mesh-dataflow.c`.
**W6. Invocation path** — worker dequeue → operand array → supplied function → native completion handler. Files: `rdma/mesh-call.c` (`mesh_call_progress`, `mesh_call_submit`), `swift/Mesh.swift` (`MeshInvocation`, `TensorFunction.prepare`).
**W7. Collective compositions** — `send/broadcast/scatter/gather/all*/reduce*` in `swift/Mesh.swift`: declaration-time only; no runtime code at all.
**W8. Caller bindings** — engine `mesh_layer.swift` and any example: the bound closures encode and return; no `waitUntilCompleted`, no presence read, no allocation.

### X — execution-shape typings (what "streaming" means at the cache line)

**X1. Presence is a dense stamp array.** `presence[instance][section]`: one word each, indexed by integers fixed at `start()`; `mesh_publish` is one store plus the L3 range walk. Ref: Monsoon presence bits; I-structures; Lamport single-writer. Check: direct row-indexed arrays on the publish path, without graph search or linked queues (I17); the stamp array's address is computed once.

**X2. Firing is a countdown, not a predicate.** each `(consumer, instance)` has `pending` initialized at `start()` to its operand count; publication decrements; zero → enqueue. Ref: Naiad occurrence/precursor counts; Realm event triggers. Check: `mesh_present`/any presence scan is absent from the runtime path; the only reader of presence at runtime is `syncOnRemoteFill` (I11).

**X3a. Indexed receive matching preserves independent producers.** `mesh_transfers_prepare` / `link_configure` realize peer-qualified transfer identities, writable receive backing and chunk-to-logical-row relationships. Each completion associates its actual registered page with the logical chunk through the canonical page table. Ref: TN3205; ledger D4/D5; JACCL SEND/RECV. Check: A-then-B and B-then-A, including interleaving from two peers at each node of a ring, produce the same logical values while all available queues continue draining. No QP per independent value, declaration-order SEND, payload staging copy or fixed logical-to-physical address is required. The relation applies to all declared edges and instances.

**X3. Preallocated operands resolve through canonical page mappings.** realize logical operand arrays, layouts, lifetimes and storage at `start()`. Receives land in posted registered backing; completion supplies the indexed mapping used by consumers and forwarding. A contiguous-typed backend consumes suitable contiguous sections, supported aliases of the same pages, or the output of an asynchronous placement operation using canonical operand storage. Ref: ledger D4/D5; Pallas BlockSpec; Pathways. Check: interleaved logical chunks reach the correct indexed and contiguous consumers; layout work introduces no global wait and is complete only for the operands it produces. Metadata reassignment and aliasing move no payload. If a backend requires materialization, account for its actual data movement; do not label that operation zero-copy or introduce a hidden operand store.

**X4. Capacity gates are hardware only.** the only conditional on a TX post is the verbs return value; no software counter of outstanding frames, credits, or window decides whether to post (RDMA-FIRST: "an implementation that gates on a computed target rather than on the hardware refusing the work has invented a throttle"). Ref: TN3205 credit flow control; NCCL proxy "if we have ops to progress, no need to block". Check: `grep -n "frames\s*[<>]=\?\|outstanding\|window" rdma/mesh-flow.c` returns only the capacity query at setup.

**X5. Reclamation is an event, not a query.** a reference count reaching zero pushes the pages onto a per-worker free list (single-producer/single-consumer ring, Disruptor); allocation for a new instance pops. Realization verifies the inFlight × bytes arena bound; when all realized instances are in use, N1 returns busy immediately. Neither allocation nor reclamation waits for readers. Ref: RCU grace period; Disruptor; Lee & Messerschmitt balance equations. Check: no `while`/`sleep`/`yield` around allocation; `start()` refuses an arena smaller than inFlight × bytes.

**X6. The partial tensor is plain data.** `TensorPart` and its C descriptor are POD: rank, logical section indices, bytes, layout and value flags. The descriptor holds no mutable runtime object; Mesh resolves backing through the canonical page table. A reduction contribution is the same type with `partial = true` (L5). Ref: DaCe memlet; ScaLAPACK descriptor. Check: `TensorPart` has no `class` reference field; `sizeof(struct mesh_section)` is a few words; descriptor copies neither retain readers nor require fixed backing addresses.

**X7. Supplied functions receive contiguous operand arrays and return.** `TensorFunction` is invoked with `(inputs: contiguous [operand], outputs: contiguous [operand], instance)` and must not read presence, wait, or allocate; its completion (return / Metal handler / Core ML handler) is the only thing that publishes. Ref: Pallas kernel refs; Active Messages handlers ("copies the data and increments the flag"). Check: I2; no mesh symbol other than the operand array is visible to the function body.

### R — segmentation recovery (cable pulls, replugs, port moves; between NFEs only)

Facts this group is written against: TB5 RDMA device names follow the interface, so a
replug can present the same peer on a new `rdma_enN`; the neighbour cache is empty after
a replug (`RDMA-RULES.md`); a send completion is not delivery (D12); UC has no repair
(transport-boundary.md); a wedged verbs call is unkillable. The operator's rule: a peer
pair that loses its link is the same two peers with a new indexed connection; they
re-establish who they are and what they will do; nothing inside an NFE is recovered.

**R1. Peer identity is independent of the link.** Signature: the pairing exchange carries `(node, boot_nonce, program_epoch)`; a re-pair with the same `node` on any device is the same peer with a new `connection_index`; `Topology.Link` gains `connection: Int`. Ref: TN3205 out-of-band metadata; `RDMA-RULES.md` versioned exchange. Check: `observe()` after a port move shows the same peer node with a new device and `connection+1`; no code path keys a peer by device name or GID.

**R2. Loss is observed at the completion queue and the control socket, never by a data-path timer.** Signature: a failed work completion or a control-socket error puts the link in `MESH_STOPPED` with the errno in `port.code`; every instance whose plan uses that link concludes `Result.link(peer:code:)` on its next `result()` (one load); its slots are released by the ordinary refcount events (no leak, X5). Ref: fail-stop; end-to-end; I12. Check: `grep -n "timer\|clock\|deadline" rdma/mesh-flow.c` hits only the pairing exchange (S4); no data-path function reads a clock; the killed-peer run (row 16) concludes within one completion.

**R3. Re-pairing is a bounded loop that runs while the bridge is up, regardless of clients** (subsumes 19j). Signature: the link controller repeats `{re-enumerate devices matching the configured peer, warm the neighbour cache (`ping6 ff02::1%iface`), re-create QP/CQ if the device context was lost, exchange (R1), post the planned receives}` with one bounded deadline per attempt and a fixed pause between attempts; the control listener is closed and reopened per attempt (no stale backlog). Ref: `RDMA-RULES.md` (bound every syscall; never SIGKILL); TN3205. Check: `link_run` has no `if(!transfers) return`; a follower started 5 minutes before rank 0 pairs when rank 0 attaches; `bin/mesh-status.sh` shows `attempt` counts.

**R4. A realized program survives a link loss.** Signature: after `MESH_STOPPED → MESH_PAIRED` on the same peer, `Mesh.submit(next)` succeeds with no new `Mesh(…)`, no `start()`, no re-registration by the client: sections, bindings, planned pages and send-edge tables are unchanged; the bridge re-posts the planned receives for the new connection from the same tables. Instances in flight at the loss are the ones that concluded in R2. Ref: MPI-4 persistent collectives (a persistent request outlives a failed start); Pathways. Check: `grep -n "start()\|realize" ` in the driver's recovery path is empty; the demonstration in R7.

**R5. Port moves are configuration-free.** Signature: the bridge config names peers, not devices: `links=('*,1,…')` or a device list per peer; at attempt time the controller selects any `PORT_ACTIVE` device whose neighbour table contains the peer's link-local address. Placement routes (T2) name pairs, never devices. Ref: `RDMA-RULES.md` "warm the neighbour cache before RTR". Check: moving the cable to another port on either node re-pairs with no config edit and `observe()` shows the new device.

**R6. Partition with P > 2 is a topology change, not an error storm.** Signature: each side's `Result.topology(lost:, gained:)` (T5) names exactly the peers it lost; instances needing a lost peer conclude failed; the driver re-realizes on `observe()`'s reachable subgraph between NFEs; ownership never migrates on its own (I9). Check: T5's cable-pull run on a ring names one link on each side and nothing else.

**R7. Demonstration and number.** Signature: on the pair, pull the cable during an NFE: the NFE concludes `Result.link` within one completion on both sides; replug (same or other port): the next `submit` completes with no process restart, no bridge exit, no U-state process; report time-to-repair (cable in → paired) and time-to-first-completed-NFE-after-repair. Check: bridge pids unchanged before/after; both numbers in `output_data/mesh_recovery/`; repeated three times.

### E — engine integration and measurement

**E1. The serving step calls Mesh at the Megatron points.** in `metal-microbench`, one file ≤ 300 lines: a Gemma-4 layer where `o_proj` and `down_proj` partials go `call(dot) → reduceScatter(using: add) → call(norm+residual) → allGather`; every other op is `call` on the rank's head/column range with existing kernels bound as `TensorFunction.metal`. Ref: Megatron f/g; Korthikanti 2022; MLX `shard_linear`; Pallas collective matmul (reuse the local kernel). Check: `grep -c "reduceScatter\|allGather" <file>` = 2 × layers; no kernel arithmetic rewritten; solo and n-node are the same binary.

**E2. Public-path measurement with the legitimacy number.** `report(model, placement) -> {T1, T(n), capability_sum, e_karp_flatt, bounds.max, verdict}` from one script through the server endpoint. Ref: Karp–Flatt; Pope et al.; Amdahl doc. Check: I13; `verdict == superior` only when `T(n) < T1` against capability sum with the last machine's contribution positive.

**E3. Depthwise chain is not slower.** the four-FFN-residual chain run through E1/E2. Check: `T(2) < T(1)` on the public path; otherwise the row stays ✗ and names the wait that caused it.

### P — packaging

**P1. Importable and run.** `import Mesh` from a Swift target outside the repo builds with `make -C rdma all` and no environment variables; `bin/mesh-bridge.sh start` is the only prerequisite; the Core ML chain runs on the pair at ABI 43. Ref: MLX `mx.distributed.init()`. Check: a fresh clone on the Mini builds and runs the example by the README command.

**P3. Stable identity, fixed path, one listener.** `rdma/Makefile`: `codesign --force --sign "$(MESH_CODESIGN_IDENTITY)" --identifier io.mesh.bridge` (default identity `Mesh Bridge`, created once per node by `bin/mesh-codesign-identity.sh` in the System keychain; ad-hoc only as a fallback that the check flags); `make -C rdma install-bridge` installs to `$(MESH_PREFIX)/bin` (`/usr/local/mesh/bin`) in place; `bin/mesh-bridge.sh` launches only from there and refuses worktree/build/scratch paths; `install.sh` adds the installed path to the Application Firewall once. Ref: Apple code signing (designated requirement stays constant across builds for a persistent identity); `socketfilterfw`. Check: I20 on both nodes; a rebuild + `mesh-bridge.sh start` produces no firewall or Local Network dialog and no new `socketfilterfw --listapps` entry.

**P2. Pushed.** `git log @{u}..HEAD` empty on both repositories at the end of every turn.

## 5. Completion matrix

✓ done (hash) · ◐ partial (what is missing) · ✗ not started. Assignment order is row
order. The library is done when every row is ✓; partial progress is reported as the
subset, never as "done".

| # | Row | Status | Evidence |
|---|---|---|---|
| 1 | S1 links list | ✓ | `c9d9e18` |
| 2 | S2 TX/RX threads per link | ✓ | `c9d9e18`; used by P1's paired Core ML run |
| 3 | S3 chunked transport invisible | ✓ | `98742c8`; P1 transported four/eight-chunk sections |
| 4 | S4 bounded pairing | ✓ | `06dcdb3`; checked nonblocking sockets, one deadline across all pairing exchanges |
| 5 | L1 partial tensor | ✓ | `swift/Mesh.swift` `TensorPart` |
| 6 | L2 supplied function | ✓ | `TensorFunction.cpu/.metal/.prediction` |
| 7 | L3 publish/fire | ✓ | `mesh-call.c` `mesh_publish`; P1 reached all eight final consumers |
| 8 | L4 ten verbs | ✓ | `Mesh.swift:279-365` |
| 9 | L6 any combine | ✓ | by construction |
| 10 | L7 lifetime | ✓ | `mesh_buffer_retain`/`mesh_collect`; used by P1 |
| 11 | L8 explicit sync + counterexample | ✓ | `examples/sync-on-remote-fill.swift` |
| 12 | P2 push both repos | ✓ | `29bb74f` mesh / `e2f99d1` engine; both remote `main` heads verified, `git log @{u}..HEAD` empty |
| P3 | stable identity, fixed install path, one listener (I20) | ◐ — CAUSE CONFIRMED | lane M unified-log evidence: macOS 26 Local Network privacy keys on the code identity; with per-build ad-hoc ids (`mesh-flow-<cdhash>`) every rebuild made the FIRST dial fail synchronously (`connect()` EHOSTUNREACH in 0.2 ms) until the grant re-cached — the 19j 'one attempt' then failed pairing (`UserEventAgent LocalNetwork: found bundle id mesh-flow-… / nehelper UUID cache miss / received prompt`). Stable `--identifier io.mesh.bridge` is the fix for that too; `mesh-codesign-identity.sh` needs interactive auth (`SecTrustSettingsSetTrustSettings: authorization denied`) on both nodes; Mini installed ad-hoc+stable-id at `/usr/local/mesh/bin/mesh-flow`; Makefile signs with `Mesh Bridge`/`io.mesh.bridge` (ad-hoc fallback), `install-bridge` target, launcher refuses worktree/build paths, identity script — merged; not yet applied on either node (bridges currently run from `mesh-wt/K` at ad-hoc identity `mesh-flow-<cdhash>`; two firewall entries exist on the laptop) |
| 13 | P1 importable; Core ML chain run on the pair at ABI 43 | ✓ | Core ML chain (F=linear 64×64 seed 73, G=ReLU, [32,64], count 8, 2 stages) ran on the pair 2026-09-15 15:06:54–15:08:54 at ABI 43, laptop `rdma_en6` ↔ Mini `rdma_en3`, `paired_links:1` within 3 s of attach; region scans on both nodes show all 8 indices' reduced sections, G outputs and received contributions matching numpy (176/176); evidence `metal-microbench/output_data/mesh_p1/` (`run_coreml.sh`, `scan_rank{0,1}_after_coreml.log`, `verify_coreml.txt`); earlier the no-model example round-tripped (lane F) |
| 14 | L5 Partial type | ✓ | `858c530` (integrates `d56e8a2` + `0468d90`): `TensorPart.partial`; one setup-only throw in `call`; sends preserve contribution identity; every completed reduction gets a distinct identity without copying, including one contribution; [source derivation](distributed-reduce.md#l5-contribution-typing); W7 re-audited; module, three existing callers and engine Mesh target build |
| N1t | the slot-free event re-arms transport: RX re-posts the RECV on each freed receive page; TX send-edge chunk cursor resets per traversal; `send_ready` is a ring — files rdma/mesh-flow.c; depends on rows 19c/19e | ✗ | `submit(inFlight + k)` on a program with one `send` completes on the receiver |
| N1r | slot release on non-submitting ranks: instance completion must not require a local `submit` when work is driven by received operands | ◐ (seed defect removed; N1/X5 open) | the submission seed and its release are deleted; declared calls and transfers already retain unfinished work, including local roots before submission. No root-arrival callback, inferred broadcast or rank case is needed. |
| 15 | N1 unbounded instances | ◐ | `count` remains finite. The `01f2727` attempt is removed in the integration: modulo slot selection violates I4 and can report busy while other slots are free; Metal constructed the next command buffer before publishing the completed output; transport was not re-armed. The N1 signature and N1t remain required. |
| 16 | N2 Result surface | ◐ | `3f151d3` plus integration: one-load `Mesh.result(index)`, per-instance success and native link/function errors at ABI 46; numerical retirement replaces per-call whole-program reference updates; all declared transfers contribute through their final chunk. Replaces `083ddbb`'s link scan and success-before-completion. Missing: remote-caller death detection, driver re-realization, N1 reuse and W1–W6 repairs. No killed-peer or performance claim. |
| 17 | E1 engine layer via Mesh | ✓ (unrun) | engine `6507370` `mesh_layer.swift` 229 lines, 2 collective points, 12 existing encoders bound, no kernel file changed; target `.build/libgemma_mesh.dylib` builds; not yet run on the pair (row 19) |
| 18 | E2 public measurement + Karp–Flatt | ◐ | script `tools/mesh/report.py` (engine `feda6a6`); **solo T1 measured** 2026-09-16 00:3x: gemma-4-E2B-it, public endpoint, 8/8 requests × 64 tokens, temp 0: median **16.175 ms/token** (61.8 tok/s), TTFT median 61.9 ms, memory guard 84%→78%→84% free, swapouts unchanged (`output_data/mesh_e3/solo_t64.json`); T(2) pending 19h/19i |
| 19 | E3 depthwise chain not slower | ✗ MEASURED INFERIOR 2026-09-16 01:31Z | first two-node decode through the library on the public path (engine `772a8cb`, mesh `7c7fee4`, gemma-4-E2B-it, E1b placement, 64 one-shot instances): solo 8×8 **T1 = 11.95 ms/token**, pair **T(2) = 181.65 ms/token** (per request 107→218 rising), TTFT 57.8 → 522.8 ms, S = 0.066, **Karp–Flatt e = 29.4**, capability sum 1.67 → `verdict: inferior` (`compare_A_8x8.out`); 8×16: T(2) = 228.5 (4/8 usable, then `mesh instances exhausted` as a clean engine_error). Leading cause by arithmetic: E1 binds ≈13 `TensorFunction.metal` per layer → ≈455 command buffers per decode step × ~0.28 ms crossing ≈ 127 ms of the ~170 ms delta (rows E1c, 19k); secondary: per-step host staging root call + broadcast (~1.3 MiB + 271 KiB), Mini memory pressure (73% → 33% free with the 4.7 GB arena + model mlock, invisible to the guard). Evidence `output_data/mesh_e3/` |
| E1c | segment granularity: three collective-delimited segments per layer | ✓ (unrun) | engine `da59157`: `segmentA/B/C` + the reduce `add`; command buffers per decode step rank 0 **483 → 168**, rank 1 441 → 161 (28 sliding × 5 + 7 full × 4/3); intermediates one page per instance per section at bind time; closures encode-and-return; 217 lines; re-measure row 19 |
| 19k | X8 mesh-side: consecutive `call`s on the same worker whose inputs are all local (no transport edge between them) are encoded into one command buffer at `start()` (Pallas pipelining / MLX ops-per-buffer batching), so caller granularity is not the only lever; presence firing stays per published section | ✗ | Mesh.swift `MeshInvocation` — live session's file |
| 19l | per-step host staging in E3 (`.cpu` root call memcpy of `input_tokens/positions/k_len/block_table/masks` ≈1.3 MiB + prompt 271 KiB, then `broadcast`) is the engine's own X3 violation (`event_fired_not_scanned` memory: "per-step host staging is the same defect"); step-indexed values are planned constants written once; only the sampled token moves per step | ✗ | E3 `925dfb3` design |
| 19h | receive-fill refusal is the gate, not a fatal: `rdma/mesh-flow.c:136` (`d61e503`) `if(error!=ENOMEM && error!=EAGAIN){…return -1;}` — `ibv_post_recv` on this stack returns **negative** errno (`-12`), so the initial receive fill of every link ends in `MESH_STOPPED code -12` and ABI 47 bridges cannot pair for any client (ABI 43 paired for the same clients earlier the same day). Fix: treat `±ENOMEM/±EAGAIN` from `link_post` as "stop posting, resume on the next completion" (X4's native refusal), never as `link_error` | ✓ | `link_post` now returns `|errno|` (this stack's `ibv_post_*` return negative errno), so `:136`/`:177` see `ENOMEM`/`EAGAIN` and treat the refusal as the end of the fill / the gate; re-pair verified 2026-09-16 18:1x: bridges `7c7fee4` ABI 47 on both nodes, `paired_links:1 phase:2` under a client (`mesh-stat` laptop pid 87957 / Mini pid 16592); lane K: `output_data/mesh_e3/bridge_laptop.log`, `mini/bridge_mini.log`, `run.sh` attempt ledger |
| 19i | arena bound vs one-shot instances: `mesh_section_create` (mesh-call.c:348-353) plans `count × ceil(bytes/64 KiB) × 4` pages per section; the E3 graph is ~3,500 pages (~55 MiB) per instance per rank, so the 1 GiB arena holds ≤ 18 instances — 8 requests × 64 tokens need ≥ 520 steps. Either N1 (instance reuse, the real fix) or `mesh_arena_pages` raised to ≥ 6 GiB on both nodes for a bounded run | ◐ bounded | arena raised to `mesh_arena_pages=229376` (3.5 GiB, registered on both nodes) ⇒ ≈ 64 one-shot E3 instances for a bounded run; the real fix stays N1; lane K attempt A: rank 1 `Mesh decode bootstrap: … Code=12 "Cannot allocate memory"` |
| 19j (→ R3/29c) | pairing has one 30 s attempt per client attach and the control listener keeps a stale backlog across attaches (`verbs_up` deadline; `if(provider->listener<0 && listener_up(…))`): a follower launched > 30 s before rank 0 never pairs (codes 60/65); a dead backlog connection gives `EPIPE` on the next exchange until the bridge restarts. Fix: bounded re-attempt while the client is attached; drain/close the listener between attaches | ✗ (cause = P3 Local Network first-dial failure; ABI 43 masked it by retrying inside attach) | lane K attempts 2–3 |
| E1b | MQA placement (per-rank head ranges, empty range = no attention/o_proj contribution on that rank) | ✓ (unrun) | engine `772a8cb`: `MeshLayerPlacement.heads/kvHeads` are per-rank arrays written explicitly by the caller; preconditions: exact cover of `[0,heads)`, kv empty iff heads empty, grouping 8 (full) / 2 (sliding); one-contribution `reduceScatter` is the same call (`reduce` with one part = `send`, zero copy on the owner); example `output_data/mesh_e3/placement_e2b_rank{0,1}.json` (full layers `[[0,8],[0,0]]`) verified against every precondition; rank 1 idles on the 7 full layers' attention |
| W1 | publish path waitless/guardless | ✗ | `mesh_publish` fixed bitmask walk ✓ but enqueues via `mesh_notice_push` CAS-retry Treiber stack (mesh.h:97-102) and `mesh_buffer_release`→`mesh_buffer_enqueue` claim check + list push (mesh-dataflow.c:125-130); [audit](w-audit-2026-09-15.md#w1) |
| W2 | receive path | ✗ | Dedicated polling, source identity resolution and `mesh_receive_assign` are permitted by the operator clarification. Distinct per-queue receive runs and page-table assignment preserve finite out-of-order sections without payload copying. Publication still enters the CAS notice/reclamation paths; reusable receives and general interleaved-chunk layout remain open. [Audit](w-audit-2026-09-15.md#w2--receive-path). |
| W3 | send path | ✗ | Page-table indexing is permitted. Remaining defects are the CAS/list publication queue, finite append-only `send_ready`, reclamation retry stack and missing rearm. Chunks of one section are currently issued consecutively per queue; broader chunk interleaving needs X3 layout support. [Audit](w-audit-2026-09-15.md#w3--send-path). |
| W4 | submit path | ✗ (push primitive only) | `mesh_calls_submit` is a fixed bitmask walk + `mesh_notice_push` CAS retry; no bound on `index` (N1 ◐); [audit](w-audit-2026-09-15.md#w4) |
| W5 | result/collect path | ✗ | `Mesh.result` is one status load. Reclamation still uses a duplicate-enqueue guard, CAS linked stack, deferred retries and bitmap allocation; no instance free-pool pop exists. Resolving actual backing through the page table is permitted. |
| W6 | invocation path | ✗ | Resolving received operands through `mesh_page[]` and selecting prepared views by page index is permitted. Notice dispatch still follows a linked stack; Metal command buffers are single-use; arbitrary fragmented inputs lack contiguous materialization. Core ML feature-name lookup must be evaluated as native interoperability, without replacing it with a slower lookup or hiding it in a wrapper. |
| W7 | collective compositions declaration-only | ✓ | audited f1ae04a: all conditionals in `send…allReduce` (Mesh.swift:324-409) run before `start()`; no runtime code; [audit](w-audit-2026-09-15.md#w7) |
| W8 | caller bindings encode-and-return | ◐ | The removed prefill wait remains forbidden; `lm_engine.swift:1909` still inserts `encodeWaitForEvent(graph.meshStepEvent, ...)` before `encodeDecodeOutput`, whose operand dependency must be declared through Mesh. Gram bindings encode or execute supplied numerical functions and return. The engine `encAttn` binding still reaches the host `attentionSplits` scan; prefill must publish declared operands consumed by decode. Indexed `MeshBindings` view selection is permitted. |
| 19a | X1 dense presence stamps | ◐ | presence store is one `atomic_fetch_or` ✓; the notice queues behind it are a CAS Treiber stack (W1 audit) — replace with start()-sized per-(producer,consumer) rings |
| 19b | X2 countdown firing | ✓ | `mesh_call_progress` mesh-call.c:160: `if(!--call->pending) mesh_call_submit(call)`; no runtime presence predicate except `mesh_sync_on_remote_fill` (I11) |
| 19c0 | X3a indexed matching for independent producers | ◐ source | Existing `mesh_transfers_prepare`, per-link/per-queue receive tables and `mesh_receive_assign` support either section publication order over finite receive runs. The operator clarification removes the invented immutable-address/QP-per-value prerequisite. General interleaving, reuse and multi-peer integration remain to be completed and demonstrated. |
| 19c | X3 canonical page-backed operands and asynchronous layout | ◐ source | `mesh_call_submit` and `MeshBindings` already resolve actual received backing. Current contiguous bindings depend on consecutive chunks of each send within one queue; multi-link/interleaved chunks need indexed sections or asynchronous contiguous placement. Preserve zero-copy mapping where applicable and expose no transport chunking to callers. |
| 19d | X4 hardware-only capacity gate | ◐ (admission implemented; W2/W3 open) | `38adbe6`: `mesh_queue.pending/capacity` and both software gates deleted; one completion poll and one available post per progress step; native refusal preserves the cursor; initial receives fill before traffic; only per-request fit checked at setup; bridge builds; source bodies and remaining defects in [audit](w-audit-2026-09-15.md#w3) |
| 19e | X5 reclamation as free-list event; arena bound at start() | ✗ | X5/W5: replace `mesh_buffer_enqueue`/`reclaim_head`/`mesh_collect`/`link_collect` + collector thread; refcount→0 pushes `(first,pages)` on a per-worker SPSC ring; `Mesh.result` one-load status word (N2) |
| 19f | X6 TensorPart is POD | ◐ (descriptor implemented; X3/W open) | `5203b2b`: primitive fields and an optional 32-byte C `mesh_section`; setup ownership ends after binding, declared uses own actual accesses. Logical indices intentionally resolve through the canonical page table. W1/W5 reclamation defects and general X3 layout integration remain. |
| 19g | X7 functions get contiguous operand arrays | ✓ (verify) | `MeshOperands` |
| 20 | G1 contraction/Gram as the same calls | ◐ (source composition complete; W1–W6 open) | `c30fe6f` `examples/gram-chain.swift` (108 lines): one supplied-function composition for `ZA`, `RᵀH`, `RU`, `YW`; direct Gram-to-down and residual-to-next-block operands; unequal rectangular tiles, explicit owners, configured depth; `make -C rdma gram-chain` builds; unrun, no performance claim; [algebra and flow](function-chain.md#g1-gram-and-projection-chain) |
| 21 | G2 indices as data (caller pattern) | ✓ | `e47f669` `examples/indexed-gather.swift` 120 lines: index sections are ordinary `TensorPart`s (produced locally or received by `send`); routed expert = `map` over `[x, w0, w1, expertIdx]`, neighbourhood sum = `map` over `[table, idx]`; selection inside the supplied function; firing by X2 countdown; library delta 0; no new symbol |
| 22 | G3 two callers | ✗ | — |
| 23 | T1 topology observed | ◐ | `3e0d603` `swift/Topology.swift` (`Topology{nodes, links: [Pair: [Link]]}`, `Mesh.observe()` via read-only `mesh_observe`, no client slot; bandwidth from `ibv_query_port`; ABI 44); missing: latency (row 22a), idle pairing so `observe()` sees cabled links before a client (row 22b); tree/ring+spur run needs T2 |
| 24 | T3 multi-link striping | ◐ on branch `row/T1-T3` (`60bf25a`, not merged) | Revisit the implementation against the corrected I17/I18 and X3: chunk-to-link placement, completion-driven fan-in and indexed backing are permitted. Its banked log, padding, hidden receive-width state, reuse and contiguous operand path still need integration review; the branch is not accepted merely by this specification correction. |
| 22a | T1 latency observation: bridge records per-link `mesh_link_info.latency` with validity; `observe()` exposes seconds or nil; no pairing-RTT substitute | ✗ | — |
| 22b | T1 idle pairing: the bridge pairs every configured link at startup (no `if(!transfers) return` in `link_run`; controllers start before the client loop) so `observe()` before `Mesh(…)` sees every cabled link | ✗ | — |
| 25 | T2 routes/forwarding | ◐ source | `34233ba`: `Placement.owners/routes` and directed `Placement.Edge`; `Mesh.send` expands caller paths at declaration and binds each relay's received section as its onward SEND source. The route table is dropped before execution. Compute workers start only for declared functions, so a relay starts none. The existing Gram caller accepts routes; `examples/gram-chain-star.json` places ranks 0/2 through rank 1, which never calls `submit`. Source expansion gives two SENDs on `0 → 1 → 2` and payload bytes per cut from distinct realized legs. W7 re-audited; W1–W6 and the multi-hop run remain open. |
| 26 | T4 bounds() | ✓ | `67b46e5`+`8f6316f` `swift/Bounds.swift` pure; `rdma/bounds-table` prints 199.34/229.01, 44.81/53.67, 359.28/389.61 |
| 27 | T5 retopology Result | ✗ | — |
| 28 | T6 replicas | ✗ | — |
| 29 | T7 lease / multi-tenant | ✗ | — |
| 29a | R1 peer identity independent of the link | ✗ | pairing exchange carries node + nonce (versioned); no connection index; `observe()` has no `connection` |
| 29b | R2 loss observed at CQ/control socket; instances conclude as `Result.link`; slots released | ◐ (exhaustion path verified) | `mesh instances exhausted` surfaced as a clean `engine_error` with the leader alive (lane M run B); link loss itself untested; `link_error` records `port.code`; `Mesh.result` reads it program-wide; per-instance conclusion and slot release on loss unverified (needs the killed-peer run) |
| 29c | R3 bounded re-pair loop while the bridge is up (subsumes 19j) | ✗ | one 30 s attempt per client attach; stale listener backlog; `link_run` returns on zero transfers |
| 29d | R4 realized program survives link loss (`submit` after re-pair, no `start()`) | ✗ | — |
| 29e | R5 port moves configuration-free (peers, not devices) | ✗ | config names `rdma_enN`; no device selection by neighbour table |
| 29f | R6 partition = T5 topology change | ✗ | needs T5 |
| 29g | R7 cable-pull/replug demonstration with time-to-repair | ✗ | — |

Rows W1–W8 are the shared waitless/guardless feature: every other row's ✓ depends on them staying ✓ (I18). Rows 19a–19g are the typings that make rows 13–19 mean streaming at the cache line rather than in prose; they are assigned before row 20. Rows 20–29 are required for the 4× M5 Ultra + 4× M4 Pro deployment and for the solver
caller; they are not optional and not "later" — they are after row 19. Rows whose files
are disjoint may be worked in parallel worktrees: {14,15,16} share `Mesh.swift`/`mesh-call.c`;
{23,24} share `mesh-flow.c`; {17},{18},{26} are independent; {13},{19} need the link
and run one at a time (`RDMA-RULES.md`: one experiment at a time).

## 5a. Lane assignments (live; edit when a lane finishes)

Rows below are being worked in parallel worktrees by steering agents driving `codex exec`.
An agent picking "the first ✗/◐ row" skips assigned rows and takes the next unassigned one.

| Rows | Lane | Worktree / branch |
|---|---|---|
| 19c, 19d, 19e, 19a, N1t (mesh runtime paths: `mesh-flow.c`, `mesh-call.c`, `mesh.h`, `Mesh.swift`) | live Codex session in `~/dox/mesh` (took 19c/19d itself at `93826c3`) | main — concurrent N1/N2 ancestry integrated; the N1 modulo path is removed and N2 uses per-instance status at ABI 46; neither lifecycle row is complete |
| 19 measured run | K (assignment recorded) | The assignment named mesh `d61e503` and engine `aad2874`. Engine `74a62ff` removes that revision's forbidden wait/stamp implementation; any run of `aad2874` does not establish E3 acceptance for current source. No runtime handle has been verified by the runtime lane. |
| 24 (redo after 19c) | — | `row/T1-T3` holds the first attempt |

Rows 15 and 16 remain open in the runtime lane. The operator clarified I17/I18: dedicated queue progress and canonical page-table matching are permitted. X3a/X3 now require completion of that indexed path and asynchronous layout, not immutable native addresses. X4 admission is implemented at 19d; X5 and X1 remain at 19e and 19a. Other open work: 19f (POD descriptor implemented; layout integration and reclamation still need 19c/19e), 20 (caller implemented; W1–W6 open), 22, 22a, 22b, 25, 27, 28, 29. Lane A's L5 work is integrated; the concurrent N1/N2 attempt has been integrated and audited as recorded in rows 15/16. Note: both bridges were restarted at 14:57 from the main checkouts (`/Users/mdot/dox/mesh/rdma/mesh-flow`, `~/mesh/rdma/mesh-flow` on the Mini, ABI 43, same config); the `mesh-wt/P1` worktree is no longer load-bearing. ABI 47 has been built locally but not deployed by the runtime lane.

## 6. Forbidden substitutions (revert on sight)

- A bibliography section, JSON record, trace field, evidence file, or "source
  inventory" without a §5 row changing status (I14, I15).
- An expression compiler, kernel zoo, scalar DSL, or any numerics inside mesh (I7).
- Migrating a caller's operators into mesh (I8), or a model/layer API in mesh.
- A cost model, auto-placement, or a verb inferred from world size (I9).
- A reuse guard, claim check, ready byte, occupancy check, or handshake (I3, I4).
- A default wait: any `waitUntilCompleted`, semaphore, or poll on the publish path (I2).
- A "gold harness", test suite, or benchmark that imports engine internals (I13, I15).
- Deleting a §5 ✓ artifact to rewrite it, unless the rewrite is itself a row.
- Revising a body of code that has not run its first deliverable instead of deleting
  it (I16). The 15,041-line deletion at `5762898` was correct and derivable in advance.
- Marking a row ✓ on the strength of a build, a deletion, or a document (I13).
- Declaring completion with any row ✗ or ◐.
