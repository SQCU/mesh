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

This document is the goal. It is handed verbatim to a Codex session, a Claude session,
or a Claude that launches `codex` as a subagent. Every row is a deliverable with a
signature, a reference to cross-implement from, and a check readable from source; the
work is finished when every row is ✓; a row that is not listed is not work; an
invariant (§1) is never a row an agent may fill in with a specialization. The
operator's other normative sentences are collected verbatim in
[collective-goals.md](collective-goals.md) and [SPECIFICATION.md](SPECIFICATION.md) §25.

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
| I17 | No pointer chase on the hot path. Every runtime structure touched between a publication and the next numerical submission is a contiguous array indexed by an integer fixed at `start()`: presence stamps, consumer ranges, pending counts, operand addresses, free lists. No linked list, hash map, tree, page-table lookup, page permutation, or software capacity counter is consulted after `start()`. The only runtime conditionals are: a pending count reaching zero, a reference count reaching zero, and the verbs call refusing a post. | `grep -n "->next\|hash\|dict\|tree\|permut\|assign" rdma/mesh-call.c rdma/mesh-dataflow.c rdma/mesh-flow.c` empty on the publish/receive/submit paths; the three conditionals above are the only `if` in those functions |
| I18 | Guardless and waitless everywhere, as a shared feature of every row. After `start()`, no function in the library — on any path, in any backend binding, in any collective composition — waits, spins, sleeps, retries, polls, or branches on data other than the three conditionals of I17. A row that introduces one anywhere fails, whatever else it delivers; a row is ✓ only if every W row (§4 group W) for the paths it touches is still ✓ after the change. | group W audit table below; each W row lists the function and its complete set of conditionals |

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

Each W row names the functions on one path and is ✓ only when their bodies contain no
wait/spin/sleep/retry/poll and no data-dependent branch beyond: pending count → 0,
reference count → 0, verbs post refused. The check is the function body itself, quoted
in the Evidence column with its conditionals enumerated. Any later commit touching a
path re-audits its W row before its own row can flip.

**W1. Publish path** — numerical completion → `mesh_publish` → consumer range walk → enqueue. Files: `rdma/mesh-call.c` (`mesh_call_complete`, `mesh_publish`), `rdma/mesh-dataflow.c`.
**W2. Receive path** — CQ completion → presence store → publish. Files: `rdma/mesh-flow.c` (`rx_thread`, receive completion handler).
**W3. Send path** — publication notice → post chunks → completion → reference release. Files: `rdma/mesh-flow.c` (`tx_thread`, post loop, send completion).
**W4. Submit path** — `Mesh.submit` → `mesh_calls_submit` → root workers. Files: `swift/Mesh.swift`, `rdma/mesh-call.c`.
**W5. Result/collect path** — `Mesh.result`, refcount → free list, `mesh_collect`. Files: `rdma/mesh-call.c`, `rdma/mesh-dataflow.c`.
**W6. Invocation path** — worker dequeue → operand array → supplied function → native completion handler. Files: `rdma/mesh-call.c` (`mesh_call_progress`, `mesh_call_submit`), `swift/Mesh.swift` (`MeshInvocation`, `TensorFunction.prepare`).
**W7. Collective compositions** — `send/broadcast/scatter/gather/all*/reduce*` in `swift/Mesh.swift`: declaration-time only; no runtime code at all.
**W8. Caller bindings** — engine `mesh_layer.swift` and any example: the bound closures encode and return; no `waitUntilCompleted`, no presence read, no allocation.

### X — execution-shape typings (what "streaming" means at the cache line)

**X1. Presence is a dense stamp array.** `presence[instance][section]`: one word each, indexed by integers fixed at `start()`; `mesh_publish` is one store plus the L3 range walk. Ref: Monsoon presence bits; I-structures; Lamport single-writer. Check: no map/list/hash on the publish path (I17); the stamp array's address is computed once.

**X2. Firing is a countdown, not a predicate.** each `(consumer, instance)` has `pending` initialized at `start()` to its operand count; publication decrements; zero → enqueue. Ref: Naiad occurrence/precursor counts; Realm event triggers. Check: `mesh_present`/any presence scan is absent from the runtime path; the only reader of presence at runtime is `syncOnRemoteFill` (I11).

**X3a. Native matching preserves independent producers.** `mesh_transfers_prepare` / `link_configure` realize a `mesh_chunk_route` table mapping `(edge, instance, chunk)` to `(channel, receiveOrdinal, page, publishRow)` within native QP capacity. Ref: TN3205; ledger D5; JACCL SEND/RECV. Check: two independent equal-framed sends, produced A-then-B or B-then-A, both land at their fixed planned addresses without withholding either ready producer, copying payload, or permuting pages. The same requirement holds for all declared edges and instances. This missing table is a prerequisite of X3; posting in declaration order does not satisfy it.

**X3. Addresses are fixed at realize; receives land where they are planned.** every operand address for every in-flight instance is computed at `start()` into a contiguous operand array; each transport chunk's RECV is posted on its planned destination page, so the page-table permutation (`mesh_receive_assign` forward/inverse exchange) and the invocation-time page resolve are deleted. The supplied function is invoked with the prebuilt operand array. Ref: ledger D4 (RECV on the consumer's pages); TensorFlow Send/Recv; Pathways "outputs sent directly into node B's input buffers". Check: `grep -n "mesh_section_page\|assign\|permut" rdma/mesh-call.c rdma/mesh-flow.c` empty on the receive and invoke paths; `mesh_transfers_prepare` allocates each chunk's page and posts its RECV there.

**X4. Capacity gates are hardware only.** the only conditional on a TX post is the verbs return value; no software counter of outstanding frames, credits, or window decides whether to post (RDMA-FIRST: "an implementation that gates on a computed target rather than on the hardware refusing the work has invented a throttle"). Ref: TN3205 credit flow control; NCCL proxy "if we have ops to progress, no need to block". Check: `grep -n "frames\s*[<>]=\?\|outstanding\|window" rdma/mesh-flow.c` returns only the capacity query at setup.

**X5. Reclamation is an event, not a query.** a reference count reaching zero pushes the pages onto a per-worker free list (single-producer/single-consumer ring, Disruptor); allocation for a new instance pops; an empty list means the arena was mis-sized at `start()` and is reported there by the SDF bound, never discovered by waiting. Ref: RCU grace period; Disruptor; Lee & Messerschmitt balance equations. Check: no `while`/`sleep`/`yield` around allocation; `start()` refuses an arena smaller than inFlight × bytes.

**X6. The partial tensor is plain data.** `TensorPart` and its C descriptor are POD: `(rank, address, bytes, stampIndex)` per instance, no reference to a mutable runtime object, no method that consults runtime state except `present` (one load). A reduction contribution is the same type with `partial = true` (L5). Ref: DaCe memlet; ScaLAPACK descriptor. Check: `TensorPart` has no `class` reference field; `sizeof(struct mesh_section)` is a few words; equality is bitwise.

**X7. Supplied functions receive contiguous operand arrays and return.** `TensorFunction` is invoked with `(inputs: contiguous [operand], outputs: contiguous [operand], instance)` and must not read presence, wait, or allocate; its completion (return / Metal handler / Core ML handler) is the only thing that publishes. Ref: Pallas kernel refs; Active Messages handlers ("copies the data and increments the flag"). Check: I2; no mesh symbol other than the operand array is visible to the function body.

### E — engine integration and measurement

**E1. The serving step calls Mesh at the Megatron points.** in `metal-microbench`, one file ≤ 300 lines: a Gemma-4 layer where `o_proj` and `down_proj` partials go `call(dot) → reduceScatter(using: add) → call(norm+residual) → allGather`; every other op is `call` on the rank's head/column range with existing kernels bound as `TensorFunction.metal`. Ref: Megatron f/g; Korthikanti 2022; MLX `shard_linear`; Pallas collective matmul (reuse the local kernel). Check: `grep -c "reduceScatter\|allGather" <file>` = 2 × layers; no kernel arithmetic rewritten; solo and n-node are the same binary.

**E2. Public-path measurement with the legitimacy number.** `report(model, placement) -> {T1, T(n), capability_sum, e_karp_flatt, bounds.max, verdict}` from one script through the server endpoint. Ref: Karp–Flatt; Pope et al.; Amdahl doc. Check: I13; `verdict == superior` only when `T(n) < T1` against capability sum with the last machine's contribution positive.

**E3. Depthwise chain is not slower.** the four-FFN-residual chain run through E1/E2. Check: `T(2) < T(1)` on the public path; otherwise the row stays ✗ and names the wait that caused it.

### P — packaging

**P1. Importable and run.** `import Mesh` from a Swift target outside the repo builds with `make -C rdma all` and no environment variables; `bin/mesh-bridge.sh start` is the only prerequisite; the Core ML chain runs on the pair at ABI 43. Ref: MLX `mx.distributed.init()`. Check: a fresh clone on the Mini builds and runs the example by the README command.

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
| 13 | P1 importable; Core ML chain run on the pair at ABI 43 | ✓ | Core ML chain (F=linear 64×64 seed 73, G=ReLU, [32,64], count 8, 2 stages) ran on the pair 2026-09-15 15:06:54–15:08:54 at ABI 43, laptop `rdma_en6` ↔ Mini `rdma_en3`, `paired_links:1` within 3 s of attach; region scans on both nodes show all 8 indices' reduced sections, G outputs and received contributions matching numpy (176/176); evidence `metal-microbench/output_data/mesh_p1/` (`run_coreml.sh`, `scan_rank{0,1}_after_coreml.log`, `verify_coreml.txt`); earlier the no-model example round-tripped (lane F) |
| 14 | L5 Partial type | ✓ | `858c530` (integrates `d56e8a2` + `0468d90`): `TensorPart.partial`; one setup-only throw in `call`; sends preserve contribution identity; every completed reduction gets a distinct identity without copying, including one contribution; [source derivation](distributed-reduce.md#l5-contribution-typing); W7 re-audited; module, three existing callers and engine Mesh target build |
| N1t | the slot-free event re-arms transport: RX re-posts the RECV on each freed receive page; TX send-edge chunk cursor resets per traversal; `send_ready` is a ring — files rdma/mesh-flow.c; depends on rows 19c/19e | ✗ | `submit(inFlight + k)` on a program with one `send` completes on the receiver |
| N1r | slot release on non-submitting ranks: instance completion must not require a local `submit` when work is driven by received operands | ◐ (seed defect removed; N1/X5 open) | the submission seed and its release are deleted; declared calls and transfers already retain unfinished work, including local roots before submission. No root-arrival callback, inferred broadcast or rank case is needed. |
| 15 | N1 unbounded instances | ◐ | `count` remains finite. The `01f2727` attempt is removed in the integration: modulo slot selection violates I4 and can report busy while other slots are free; Metal constructed the next command buffer before publishing the completed output; transport was not re-armed. The N1 signature and N1t remain required. |
| 16 | N2 Result surface | ◐ | `3f151d3` plus integration: one-load `Mesh.result(index)`, per-instance success and native link/function errors at ABI 46; numerical retirement replaces per-call whole-program reference updates; all declared transfers contribute through their final chunk. Replaces `083ddbb`'s link scan and success-before-completion. Missing: remote-caller death detection, driver re-realization, N1 reuse and W1–W6 repairs. No killed-peer or performance claim. |
| 17 | E1 engine layer via Mesh | ✓ (unrun) | engine `6507370` `mesh_layer.swift` 229 lines, 2 collective points, 12 existing encoders bound, no kernel file changed; target `.build/libgemma_mesh.dylib` builds; not yet run on the pair (row 19) |
| 18 | E2 public measurement + Karp–Flatt | ◐ | script `tools/mesh/report.py` (engine `feda6a6`); **solo T1 measured** 2026-09-16 00:3x: gemma-4-E2B-it, public endpoint, 8/8 requests × 64 tokens, temp 0: median **16.175 ms/token** (61.8 tok/s), TTFT median 61.9 ms, memory guard 84%→78%→84% free, swapouts unchanged (`output_data/mesh_e3/solo_t64.json`); T(2) pending 19h/19i |
| 19 | E3 depthwise chain not slower | ◐ source; prefill integration open | engine `925dfb3`+`8c1e27b` retain decode through Mesh. `769567d`+`aad2874` are reverted by `74a62ff`: they inserted an implicit `encodeWaitForEvent(prefillDone)` inside the supplied embed function, copied prompt storage and host-polled stamps, contradicting I4/I18 and §6. Prefill must publish operands consumed by decode through the declared Mesh dataflow. N1 reuse, N2 driver recovery and prompt-derived KV on all ranks remain required before acceptance. Existing E1 bindings require `LM_PREFILL_BACKEND=mps\|tensor` and `LM_MATRIX_LAYOUT=nt`; `encAttn→attentionSplits` still scans on the host inside the bound closure. No speedup has been demonstrated. |
| 19h | receive-fill refusal is the gate, not a fatal: `rdma/mesh-flow.c:136` (`d61e503`) `if(error!=ENOMEM && error!=EAGAIN){…return -1;}` — `ibv_post_recv` on this stack returns **negative** errno (`-12`), so the initial receive fill of every link ends in `MESH_STOPPED code -12` and ABI 47 bridges cannot pair for any client (ABI 43 paired for the same clients earlier the same day). Fix: treat `±ENOMEM/±EAGAIN` from `link_post` as "stop posting, resume on the next completion" (X4's native refusal), never as `link_error` | ◐ source fix, re-pair test pending | `link_post` now returns `|errno|` (this stack's `ibv_post_*` return negative errno), so `:136`/`:177` see `ENOMEM`/`EAGAIN` and treat the refusal as the end of the fill / the gate; lane K: `output_data/mesh_e3/bridge_laptop.log`, `mini/bridge_mini.log`, `run.sh` attempt ledger |
| 19i | arena bound vs one-shot instances: `mesh_section_create` (mesh-call.c:348-353) plans `count × ceil(bytes/64 KiB) × 4` pages per section; the E3 graph is ~3,500 pages (~55 MiB) per instance per rank, so the 1 GiB arena holds ≤ 18 instances — 8 requests × 64 tokens need ≥ 520 steps. Either N1 (instance reuse, the real fix) or `mesh_arena_pages` raised to ≥ 6 GiB on both nodes for a bounded run | ✗ | lane K attempt A: rank 1 `Mesh decode bootstrap: … Code=12 "Cannot allocate memory"` |
| 19j | pairing has one 30 s attempt per client attach and the control listener keeps a stale backlog across attaches (`verbs_up` deadline; `if(provider->listener<0 && listener_up(…))`): a follower launched > 30 s before rank 0 never pairs (codes 60/65); a dead backlog connection gives `EPIPE` on the next exchange until the bridge restarts. Fix: bounded re-attempt while the client is attached; drain/close the listener between attaches | ✗ | lane K attempts 2–3 |
| E1b | MQA placement: gemma-4-E2B has 8 Q heads / 1 KV head on every layer; `mesh_layer.swift:43-45` force full layers to `heads [0,8]` on both ranks (grouping of 8) so the 7 full layers' o_proj partials are reduce-summed twice — a numerics defect. Fix: allow an empty head range on a rank (that rank contributes no o_proj partial; `reduceScatter` with one contribution), so full-layer attention runs whole on one rank | ✗ | lane K placement analysis |
| W1 | publish path waitless/guardless | ✗ | `mesh_publish` fixed bitmask walk ✓ but enqueues via `mesh_notice_push` CAS-retry Treiber stack (mesh.h:97-102) and `mesh_buffer_release`→`mesh_buffer_enqueue` claim check + list push (mesh-dataflow.c:125-130); [audit](w-audit-2026-09-15.md#w1) |
| W2 | receive path | ✗ | X4 removes software admission count/gate; each progress step polls one completion and attempts one post before publication. The sticky `failed` latch and publication guard are deleted: native post/poll/completion failure publishes `Result` and ends the affected link invocation; the controller no longer re-pairs the failed invocation. Remaining: sequential receive pages, source-tag read, per-source cursor and `mesh_receive_assign` permutation; X3a matching is a prerequisite for fixed receives; [audit](w-audit-2026-09-15.md#w2) |
| W3 | send path | ✗ | X4 removes the software admission count/gate; remaining: `send_ready` indirection, per-chunk `mesh_page[]` load, `->next` notice walk, reclamation retry stack; [audit](w-audit-2026-09-15.md#w3) |
| W4 | submit path | ✗ (push primitive only) | `mesh_calls_submit` is a fixed bitmask walk + `mesh_notice_push` CAS retry; no bound on `index` (N1 ◐); [audit](w-audit-2026-09-15.md#w4) |
| W5 | result/collect path | ✗ | `Mesh.result` is one status load; reclaim remains a polled linked-list stack (`link_collect` thread spins on `mesh_collect`, deferral = retry, page-table lookup per block, frees into bitmap first-fit allocator) — no free-list pop exists; [audit](w-audit-2026-09-15.md#w5) |
| W6 | invocation path | ✗ | `mesh_call_submit` resolves remote operands through `mesh_page[]` at invocation (mesh-call.c:141-142); `MeshBindings.index` for received parts is arithmetic on the runtime (permuted) page (Mesh.swift:288); `MeshFeatures.featureValue` is a String-keyed Dictionary lookup during Core ML prediction (:26-28); `->next` notice walk (:157); [audit](w-audit-2026-09-15.md#w6) |
| W7 | collective compositions declaration-only | ✓ | audited f1ae04a: all conditionals in `send…allReduce` (Mesh.swift:324-409) run before `start()`; no runtime code; [audit](w-audit-2026-09-15.md#w7) |
| W8 | caller bindings encode-and-return | ✓ (contingent on W6 fix of Mesh.swift:288) | audited 1c2b5d4: every bound closure encodes and returns; `for j in owners`/`if writeKV` are start()-time constants; G1 `c30fe6f` adds three supplied Accelerate closures with no conditionals, waits, presence reads or allocations; [audit](w-audit-2026-09-15.md#w8) |
| 19a | X1 dense presence stamps | ◐ | presence store is one `atomic_fetch_or` ✓; the notice queues behind it are a CAS Treiber stack (audit item 5) — replace with start()-sized per-(producer,consumer) rings |
| 19b | X2 countdown firing | ✓ | `mesh_call_progress` mesh-call.c:160: `if(!--call->pending) mesh_call_submit(call)`; no runtime presence predicate except `mesh_sync_on_remote_fill` (I11) |
| 19c0 | X3a realized native matching for independent producers | ✗ | missing `mesh_chunk_route` table; current `mesh_transfers_prepare` binds pages but not native receive matching; [A/B counterexample](w-audit-2026-09-15.md#x3a-receive-matching-prerequisite) shows why declaration-order SEND adds a dependency |
| 19c | X3 addresses fixed at realize; RECV on the planned page; permutation deleted | ✗ | needs X3a before W2/W3/W6 mapping deletion; the current page bindings alone do not match arbitrary independent publication order; then remove permutation, tag/cursor mapping, send indirection and invocation-time resolution |
| 19d | X4 hardware-only capacity gate | ◐ (admission implemented; W2/W3 open) | `38adbe6`: `mesh_queue.pending/capacity` and both software gates deleted; one completion poll and one available post per progress step; native refusal preserves the cursor; initial receives fill before traffic; only per-request fit checked at setup; bridge builds; source bodies and remaining defects in [audit](w-audit-2026-09-15.md#w3) |
| 19e | X5 reclamation as free-list event; arena bound at start() | ✗ | fix = audit item 4: delete `mesh_buffer_enqueue`/`reclaim_head`/`mesh_collect`/`link_collect` + collector thread; refcount→0 pushes `(first,pages)` on a per-worker SPSC ring; `Mesh.result` one-load status word (N2) |
| 19f | X6 TensorPart is POD | ◐ (descriptor implemented; X3/W open) | `5203b2b`: `TensorPart` now contains primitive fields and an optional 32-byte C `mesh_section`; `MeshSection` and its ARC lifetime are deleted. The C descriptor owns the receive-channel field. Setup owns allocated sections until bindings are realized; numerical and transport uses then own their accesses. ABI 47 deletes the retain retry/rollback and sealed protocol. Fixed per-instance addresses still require X3; W1/W5 reclamation defects remain. |
| 19g | X7 functions get contiguous operand arrays | ✓ (verify) | `MeshOperands` |
| 20 | G1 contraction/Gram as the same calls | ◐ (source composition complete; W1–W6 open) | `c30fe6f` `examples/gram-chain.swift` (108 lines): one supplied-function composition for `ZA`, `RᵀH`, `RU`, `YW`; direct Gram-to-down and residual-to-next-block operands; unequal rectangular tiles, explicit owners, configured depth; `make -C rdma gram-chain` builds; unrun, no performance claim; [algebra and flow](function-chain.md#g1-gram-and-projection-chain) |
| 21 | G2 indices as data (caller pattern) | ✓ | `e47f669` `examples/indexed-gather.swift` 120 lines: index sections are ordinary `TensorPart`s (produced locally or received by `send`); routed expert = `map` over `[x, w0, w1, expertIdx]`, neighbourhood sum = `map` over `[table, idx]`; selection inside the supplied function; firing by X2 countdown; library delta 0; no new symbol |
| 22 | G3 two callers | ✗ | — |
| 23 | T1 topology observed | ◐ | `3e0d603` `swift/Topology.swift` (`Topology{nodes, links: [Pair: [Link]]}`, `Mesh.observe()` via read-only `mesh_observe`, no client slot; bandwidth from `ibv_query_port`; ABI 44); missing: latency (row 22a), idle pairing so `observe()` sees cabled links before a client (row 22b); tree/ring+spur run needs T2 |
| 24 | T3 multi-link striping | ◐ on branch `row/T1-T3` (`60bf25a`, not merged) | codex striped chunk c → link c mod L at realization but did it with a banked append log walked by per-link cursors, a per-operand `remaining` fan-in counter, padding chunks and hidden `receive_width` state — runtime structures I17 forbids and that W2/W3's deletions remove; re-do on top of row 19c as a start()-time column in the `send_edge` table (chunk → link, planned page per link), then merge |
| 22a | T1 latency observation: bridge records per-link `mesh_link_info.latency` with validity; `observe()` exposes seconds or nil; no pairing-RTT substitute | ✗ | — |
| 22b | T1 idle pairing: the bridge pairs every configured link at startup (no `if(!transfers) return` in `link_run`; controllers start before the client loop) so `observe()` before `Mesh(…)` sees every cabled link | ✗ | — |
| 25 | T2 routes/forwarding | ◐ source | `34233ba`: `Placement.owners/routes` and directed `Placement.Edge`; `Mesh.send` expands caller paths at declaration and binds each relay's received section as its onward SEND source. The route table is dropped before execution. Compute workers start only for declared functions, so a relay starts none. The existing Gram caller accepts routes; `examples/gram-chain-star.json` places ranks 0/2 through rank 1, which never calls `submit`. Source expansion gives two SENDs on `0 → 1 → 2` and payload bytes per cut from distinct realized legs. W7 re-audited; W1–W6 and the multi-hop run remain open. |
| 26 | T4 bounds() | ✓ | `67b46e5`+`8f6316f` `swift/Bounds.swift` pure; `rdma/bounds-table` prints 199.34/229.01, 44.81/53.67, 359.28/389.61 |
| 27 | T5 retopology Result | ✗ | — |
| 28 | T6 replicas | ✗ | — |
| 29 | T7 lease / multi-tenant | ✗ | — |

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

Rows 15 and 16 remain open in the runtime lane. X3a matching at 19c0 precedes X3 at 19c; X4 admission is implemented at 19d; X5 and X1 remain at 19e and 19a. Other open work: 19f (POD descriptor implemented; fixed addresses and reclamation still need 19c/19e), 20 (caller implemented; W1–W6 open), 22, 22a, 22b, 25, 27, 28, 29. Lane A's L5 work is integrated; the concurrent N1/N2 attempt has been integrated and audited as recorded in rows 15/16. Note: both bridges were restarted at 14:57 from the main checkouts (`/Users/mdot/dox/mesh/rdma/mesh-flow`, `~/mesh/rdma/mesh-flow` on the Mini, ABI 43, same config); the `mesh-wt/P1` worktree is no longer load-bearing. ABI 47 has been built locally but not deployed by the runtime lane.

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
