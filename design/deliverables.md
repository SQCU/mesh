# Deliverables — the completion matrix for the streaming partial-tensor collective library

This document is the goal. It is written to be handed verbatim to a Codex session, a
Claude session, or a Claude that launches `codex` as a subagent. It is complete in the
sense that: every row is a deliverable with a signature, a reference to cross-implement
from, and a check readable from source; the work is finished when every row is ✓; a
row that is not listed is not work; and an invariant (§1) is never a row an agent may
fill in with a specialization.

Normative sentences are the operator's, collected verbatim in
[collective-goals.md](collective-goals.md) and [SPECIFICATION.md](SPECIFICATION.md) §25.
Everything below is derived from those and from source; no paper adds a requirement.

## 0. How to use this document as a goal

For a steering agent (Claude launching Codex, or a Codex goal continuation):

1. Read §5 (the matrix). Take the **first row whose status is ✗ or ◐**, in order.
2. The assignment is exactly that row's *Artifact*. Produce it in the named file(s),
   using the named reference; nothing else changes.
3. Run the row's *Check*. If it holds, set the row to ✓ with the commit hash, commit
   with the row id in the subject, **push**, stop the turn.
4. If the row cannot be done without a symbol that does not exist, do not invent it:
   write the missing row into §5 as a new ✗ line *above* the current one, with a
   signature and a check, and stop the turn. That is progress; a wrapper is not.
5. Anything in §6 (forbidden substitutions) is reverted on sight, regardless of who
   committed it.

For an implementing agent, the same procedure without the choice: one row per turn.
"Continue working toward the goal" means "take the next row", never "improve the code".

## 1. Invariants (never rows; violations are defects wherever they appear)

| id | Invariant | Check (source, not runs) |
|---|---|---|
| I1 | Two-sided only. `IBV_WR_SEND` into a posted RECV; never WRITE/READ/atomics (TN3205; hellas `-ENOTSUPP`). | `grep -n "IBV_WR_RDMA\|rkey\|remote_addr" rdma/` empty |
| I2 | Producers never wait. Publication happens when the writes are visible and returns. | no `wait`/`semaphore`/`poll` on the path from numerical completion to `mesh_publish` |
| I3 | Presence = receive completion. No acknowledgement, handshake, health check, credit protocol, or in-band sentinel exists in the data path. | `grep -n "ack\|handshake\|ready byte\|sentinel\|magic" rdma/mesh-flow.c rdma/mesh-call.c` empty outside the one-time out-of-band pairing |
| I4 | Memory mapping, not control flow. Every simultaneously live value has its own registered pages; no reuse guard, claim check, occupancy check or slot-modulo. | `grep -n "claim\|EBUSY\|occup\|modulo\|% *count" rdma/mesh-call.c rdma/mesh-dataflow.c` empty |
| I5 | Lifetime is reference counting owned by mesh; callers never free. Release returns pages to the pool without zeroing. | no public `free`/`release` in `swift/Mesh.swift` for values; `mesh_collect` has no `memset` |
| I6 | RDMA TX and RX each own a hardware thread per link and share it with nothing. All queues drain and refill immediately. | one `pthread_create` per link per direction in `rdma/mesh-flow.c`; no numerical call on those threads |
| I7 | Mesh contains no numerics. Matmul, activation, norm, sum, max, attention, sampling are supplied `TensorFunction`s. | `grep -niE "matmul|gelu|softmax|rmsnorm|vDSP|MPSMatrix" swift/Mesh.swift rdma/*.c` empty |
| I8 | Mesh contains no application. No model, layer, game, policy, or caller name appears in the library. | `grep -rniE "gemma|xonotic|havocbot|layer|residual|policy" swift/Mesh.swift rdma/` empty |
| I9 | Caller placement only. No cost model, no automatic sharding, no topology inference, no collective inferred from rank count or tensor type. | `grep -niE "cost|estimate|choose|infer|auto" swift/Mesh.swift` empty; `scatter`/`gather` keep explicit destinations at every world size |
| I10 | Transport fragmentation is invisible. A tensor section's bytes are not bounded by one request; chunking changes no shape, partition, function, or verb. | no public capacity query; `mesh_section_create` takes bytes, not chunks |
| I11 | Explicit synchronization exists only as a diegetic library call (`syncOnRemoteFill`) and is never a default. | the only caller of `mesh_sync_on_remote_fill` outside examples is `Mesh.syncOnRemoteFill` |
| I12 | Failures are `Result`/status values out of band; an operation concludes early on link error; the driver re-invokes at the data layer. | no `exit`/`abort` on a verbs error; `mesh_call_fail` records, never blocks |
| I13 | Amdahl superiority is the only legitimacy, measured on the public path. Compilation, deletion, documentation, and examples are not evidence. | every performance claim cites a run through the server endpoint with solo and n-node numbers and Karp–Flatt e |
| I14 | Every public construct cites one published mechanism, once, in `algorithm-sources.md`. Bibliography growth is not progress. | `algorithm-sources.md` has one section per symbol in §2, no per-commit sections |
| I15 | No tests, gold harnesses, evidence JSON, provenance records, or trace exports. Verification is source reading plus §5 checks plus public measurements. | `measurements/` gains only public-endpoint runs; no `*-provenance.json` |

## 2. Public surface (closed)

The importable library is the Swift module `Mesh` (`swift/Mesh.swift`) over the C
runtime (`rdma/mesh-call.c`, `mesh-dataflow.c`, `mesh-flow.c`, `mesh-verbs.h`, `mesh.h`).
Its surface is exactly:

```
Mesh(region:, rank:, size:, workers:, count:)            // §5 D14 changes `count` to unbounded instances
Mesh.tensor(on:sections:) -> [TensorPart]                 // a partial tensor = list of contiguous sections, each with a rank
Mesh.constant(on:bytes:initialize:) -> TensorPart
TensorFunction.cpu / .metal(device) / .prediction(model) // supplied numerics; input/output view factories
Mesh.call(function, inputs:, outputs:, on:, worker:)
Mesh.map(function, inputs: [[TensorPart]], outputs:, …)
Mesh.send(part, to:, queue:) -> TensorPart
Mesh.broadcast / scatter / allScatter / gather / allGather / allToAll
Mesh.reduce(parts, to:, using: combine) / reduceScatter(contributions, to: owners, using:) / allReduce(…)
Mesh.start()                                              // realize: storage, bindings, receives, links
Mesh.submit(index)                                        // one value instance through the realized graph
Mesh.syncOnRemoteFill(parts, index:)                      // explicit, diegetic, never default
```

Adding a symbol is a change to this section first (a ✗ row in §5), then code. A symbol
not listed here is not part of the library and is deleted with its callers.

## 3. Reference implementations to cross-implement from (not to cite around)

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

Each: **Signature** (as it must read in `swift/Mesh.swift` or the named caller) ·
**Reference** · **Check** · **Not it** (specializations that do not count).

### Group S — substrate

**S1. Links are a configured list; a bridge serves them all.**
Signature: `etc/bridge-*.conf` `links=('device,peer,local,remote[,service]' …)`; `mesh_peer_channel(ctx, peer, queue)`.
Reference: JACCL hostfile `rdma[]`. Check: no `peer=` scalar; `grep -n "size == 2\|rank == 1 - " swift rdma` empty.
Not it: a hostfile that assumes full mesh.

**S2. TX/RX per link on dedicated threads; drain and refill immediately.**
Signature: `rdma/mesh-flow.c`: `tx_thread(link)`, `rx_thread(link)`; RX reposts before delivering.
Reference: NCCL proxy; MSCCL++ PortChannel; JACCL per-wire polling. Check: I6.
Not it: a single progress loop that also runs numerical callbacks.

**S3. Chunked transport, invisible.** Signature: `mesh_section_create(ctx, bytes, count, receive, &section)`; internal `C`, `K = ceil(N/C)`.
Reference: TN3205 frame matching; llama.cpp PR #26421 (128 KiB stride cost). Check: I10.
Not it: a public "max message" or caller-side repartitioning.

**S4. Out-of-band pairing once; versioned exchange; bounded every syscall.**
Signature: `verbs_up(link)` with `XMAGIC+MESH_VERSION`; every `accept/connect/read` has a deadline.
Reference: TN3205 (GID/QPN out of band); `RDMA-RULES.md`. Check: `grep -n "accept(\|connect(\|read(" rdma/mesh-verbs.h` each followed by a timeout.
Not it: retry loops in the data path.

### Group L — library core

**L1. Partial tensor = sections with ranks.** Signature: `TensorPart(rank, bytes)`; `Mesh.tensor(on:sections:)`.
Reference: Pallas BlockSpec, ScaLAPACK descriptor. Check: a section is contiguous locally and addressable globally by `(rank, section)`; no dense hidden copy (I4).

**L2. Supplied function over sections.** Signature: `TensorFunction.cpu/.metal/.prediction`; `Mesh.call/map`.
Reference: Pallas `pallas_call` (kernel/ref), StarPU codelets. Check: I7; a CPU, a Metal, and a Core ML function each bound through the same `call`.
Not it: a mesh expression compiler; a "kernel zoo".

**L3. Publish on visibility, fire on presence.** Signature: numerical completion → `mesh_publish(row)` → consumers whose operands are all present are submitted; nothing else runs.
Reference: Monsoon; Realm events; TensorFlow Send/Recv nodes. Check: I2, I3; `mesh_publish` visits only that row's declared uses (no function scan).

**L4. Ten collective verbs as movement + supplied combine.** Signature: §2 list; `reduce*` take `using: TensorFunction`.
Reference: MPI-4.1 ch. 5 relations; Rabenseifner/Patarasuk–Yuan for `allReduce = allGather ∘ reduceScatter`; Gloo. Check: every verb is `send` + `call(combine)` compositions; movement does no arithmetic; explicit destinations at every world size (I9).
Not it: `allReduce` implemented as gather-everything-then-sum; any verb inferred from `size`.

**L5. Partial is a type.** Signature: `TensorPart.partial: Bool` set by `reduceScatter/reduce` contributions; `Mesh.call` throws `MeshError.partialOperand(part)` at setup when a non-combine function binds a partial.
Reference: DTensor `Partial`; Legion `reduce`; Korthikanti 2022. Check: one `throw` in `call`; no runtime state.
Status: ✗.

**L6. Combine is any associative supplied function.** Signature: `reduce(…, using: TensorFunction)`; the tree is binary and explicit.
Reference: `MPI_Op`; Blelloch. Check: sum/max/product/logdet all pass through the same tree with no special case for `+`. Status: ✓ by construction.

**L7. Lifetime without explicit free.** Signature: `mesh_buffer_retain/release` internal; declared uses count down on numerical completion, TX completion, object destruction; `mesh_collect` returns pages unzeroed.
Reference: RCU grace period; Disruptor. Check: I5.

**L8. Explicit sync as a library call with its counterexample.** Signature: `Mesh.syncOnRemoteFill(parts, index:)`; `examples/sync-on-remote-fill.swift` modes `parallel|serial|deadlock`.
Reference: operator quote (collective-goals.md). Check: I11. Status: ✓.

### Group G — genericity

**G1. Contraction is a two-operand `call`; K-partials are `reduceScatter` inputs.**
Signature: caller composes `call(dot, inputs:[x_k, w_k], outputs:[c_k])` per K-panel and `reduceScatter([[c_k]…], to: owners, using: add)`.
Reference: SUMMA. Check: a Gram `R(RᵀZ)` and an FFN down-projection are the same calls with different operands; the Gram output is an operand of the next `call` with no copy.
Not it: `nn.linear` inside mesh.

**G2. Indices as data (gather/scatter by a received index section).**
Signature: `Mesh.map(function, inputs: [[x], [idx]], …)` where `idx` is a `TensorPart` whose presence gates the map; the supplied function performs the indexed read.
Reference: Pallas scalar prefetch; `jax.lax.gather`; Megablox. Check: routed-expert and neighbourhood callers exist outside mesh using only `map`; no routing code in mesh (I8). Status: ✗ (routing machinery deleted; the caller-side pattern is not yet demonstrated).

**G3. Two callers, zero specialization.** Signature: `metal-microbench` and `xonotic/solver` both `import Mesh`; mesh imports neither.
Reference: StarPU/Legion. Check: I8; both callers build against the same module. Status: ✗ (no caller references `Mesh` today).

### Group N — instances and lifecycle

**N1. Realize once, submit per instance, unbounded.** Signature: `Mesh(…, count:)` becomes `Mesh(…, inFlight:)`; `submit(index)` accepts any index; storage for an instance is reclaimed by L7 and reused only after its last reader, with no guard on the caller side.
Reference: MPI-4 persistent collectives; Pathways; CUDA graphs. Check: `submit(count + 1)` is legal; `realize()` prints bytes per in-flight instance; no allocation inside `submit`.
Status: ◐ ("finite index extent cannot be rearmed").

**N2. Early conclusion as a value.** Signature: `Mesh.result(index) -> Result<Void, MeshError>`; `MeshError.link(peer, code)`, `.function(call, code)`, `.partialOperand(part)`.
Reference: fail-stop; end-to-end argument. Check: I12; the Core ML chain concludes early on a killed peer and the driver re-realizes. Status: ◐ (failures recorded; no `Result` surface).

### Group T — topology and fleet

**T1. Topology is a graph value, observed.** Signature: `Topology(nodes:, links: [(a,b): [Link(device, bandwidth, latency)]])`; `Mesh.observe() -> Topology` from live pairings.
Reference: JACCL hostfile as the full-mesh special case. Check: tree, ring+spurs, full mesh run the same caller with only the config changed; no code path assumes `links[(a,b)]` exists. Status: ◐ (links list; no observation, no graph type).

**T2. Routes belong to placement; mesh forwards on presence.** Signature: `Placement(owners:, routes: [(src,dst): [hop…]])`; `send(part, to:)` on a non-adjacent pair becomes one SEND per hop with canonical intermediate pages.
Reference: Patarasuk–Yuan (spanning tree suffices); Pathways; Monsoon. Check: leaf→leaf on a star is two SENDs with no hub wait; per-cut bytes equal the placement's declared traffic. Status: ✗.

**T3. Multi-link striping per pair.** Signature: `send(part, to:, queue:)` where `queue` selects among a pair's configured links; chunks of one section round-robin when `queue == .all`.
Reference: JACCL `ring_impl` wires; TN3205 10 QPs/device. Check: `report` shows equal bytes per link of a pair. Status: ✗.

**T4. Bounds before runs.** Signature: `bounds(program, placement, topology) -> {compute, memory, cut: {cut: s}, path: s, max}`, pure.
Reference: `docs/amdahl_superiority.md` (four lower bounds); Hockney; LogGP; COSMA. Check: reproduces the ten-minute table for 1+3, 2+17, 1+1; every report prints `measured / bounds.max`. Status: ✗.

**T5. Retopology is a `Result`; the driver re-realizes.** Signature: `MeshError.topology(lost:, gained:)`; driver: `observe(); Mesh(topology, placement.restrict(topology)).start()`.
Reference: fail-stop; Dean–Ghemawat; `RDMA-RULES.md` accessibility. Check: pull one cable mid-run on a ring: early conclusion names the link; re-realized program completes on the spanning tree; re-plug is visible to the next `observe()`. Status: ✗.

**T6. Replicas are placement, not protocol.** Signature: `Placement(…, replicas: [section: [rank, rank]])`; readers take the first present.
Reference: Legion physical instances; end-to-end argument. Check: no ack/retry/health code added; losing a link changes `bounds()` not correctness. Status: ✗.

**T7. Multi-tenant fabric by lease.** Signature: bridge serves N clients; `Mesh(…, lease: Lease(qpsPerLink:, arenaBytes:))`; setup returns errno when the lease exceeds `max_qp`/arena.
Reference: NCCL communicator per job; PMIx allocation. Check: two programs from two users run on disjoint QPs/arena; `kill -TERM` of one leaves the other's transfers untouched; no `SIGKILL` path. Status: ✗ (`hdr.client` is a single CAS slot).

### Group E — engine integration and measurement

**E1. The serving step calls Mesh at the Megatron points.** Signature (in `metal-microbench`, one file, ≤ 300 lines): a Gemma-4 layer where `o_proj` and `down_proj` partials go `call(dot) → reduceScatter(using: add) → call(norm+residual) → allGather`; every other op is `call` on the rank's head/column range with existing kernels bound as `TensorFunction.metal`.
Reference: Megatron f/g; Korthikanti 2022; MLX `shard_linear`; Pallas collective matmul (reuse the local kernel). Check: `grep -c "reduceScatter\|allGather" <file>` = 2 × layers; no kernel arithmetic rewritten; solo and n-node are the same binary. Status: ✗.

**E2. Public-path measurement with the legitimacy number.** Signature: `report(model, placement) -> {T1, T(n), capability_sum, e_karp_flatt, bounds.max, verdict}` produced by one script through the server endpoint.
Reference: Karp–Flatt; Pope et al.; Amdahl doc. Check: I13; `verdict == superior` only when `T(n) < T1` against capability sum with the last machine's contribution positive. Status: ✗.

**E3. Depthwise chain is not slower.** Signature: the four-FFN-residual chain from the operator quote, run through E1/E2. Check: `T(2) < T(1)` on the public path; otherwise the row stays ✗ and names the wait that caused it. Status: ✗.

### Group P — packaging

**P1. Importable.** Signature: `import Mesh` in a Swift package/target outside the repo builds with `make -C rdma all` products and no environment variables; `bin/mesh-bridge.sh start` is the only prerequisite.
Reference: MLX `mx.distributed.init()`. Check: a fresh clone on the Mini builds the Core ML chain example by the README command. Status: ◐ (module exists; engine Makefile integration deleted; ABI 43 bridge required and undeployed).

**P2. Pushed.** Check: `git log @{u}..HEAD` empty on both repositories at the end of every turn. Status: ✗ (mesh 80, engine 37 unpushed at 2026-09-15 13:52).

## 5. Completion matrix

Status: ✓ done (hash) · ◐ partial (what is missing) · ✗ not started. Order is the
assignment order. The library is done when every row is ✓; partial progress is any
subset of rows and is reported as that subset, never as "done".

| # | Row | Status | Evidence |
|---|---|---|---|
| 1 | S1 links list | ✓ | `c9d9e18` |
| 2 | S2 TX/RX threads per link | ✓ (unrun on RDMA) | `c9d9e18`; async-collectives.md "not been run or deployed" |
| 3 | S3 chunked transport invisible | ✓ (unrun) | `98742c8` |
| 4 | S4 bounded pairing | ◐ verify each syscall deadline | — |
| 5 | L1 partial tensor | ✓ | `swift/Mesh.swift` `TensorPart` |
| 6 | L2 supplied function | ✓ | `TensorFunction.cpu/.metal/.prediction` |
| 7 | L3 publish/fire | ✓ (unrun) | `mesh-call.c` `mesh_publish` |
| 8 | L4 ten verbs | ✓ | `Mesh.swift:279-365` |
| 9 | L6 any combine | ✓ | by construction |
| 10 | L7 lifetime | ✓ (unrun) | `mesh_buffer_retain`/`mesh_collect` |
| 11 | L8 explicit sync + counterexample | ✓ | `examples/sync-on-remote-fill.swift` |
| 12 | **P2 push both repos** | ✗ | — |
| 13 | **P1 run the Core ML chain on the pair with ABI 43** | ◐ | first RDMA run of the rewritten library |
| 14 | **L5 Partial type** | ✗ | — |
| 15 | **N1 unbounded instances** | ◐ | `count` finite |
| 16 | **N2 Result surface** | ◐ | — |
| 17 | **E1 engine layer via Mesh** | ✗ | — |
| 18 | **E2 public measurement + Karp–Flatt** | ✗ | — |
| 19 | **E3 depthwise chain not slower** | ✗ | — |
| 20 | G1 contraction/Gram as same calls | ◐ (FFN shown, Gram not) | `examples/coreml-chain.swift` |
| 21 | G2 indices as data (caller pattern) | ✗ | — |
| 22 | G3 two callers | ✗ | — |
| 23 | T1 topology observed | ◐ | links only |
| 24 | T3 multi-link striping | ✗ | — |
| 25 | T2 routes/forwarding | ✗ | — |
| 26 | T4 bounds() | ✗ | — |
| 27 | T5 retopology Result | ✗ | — |
| 28 | T6 replicas | ✗ | — |
| 29 | T7 lease / multi-tenant | ✗ | — |

Rows 12–19 are the current assignment sequence. Rows 20–29 are required for the
4× M5 Ultra + 4× M4 Pro deployment and the solver caller; they are not optional and
are not "later" — they are after row 19.

## 6. Forbidden substitutions (revert on sight)

These have each been committed before as "progress" and are not:

- A bibliography section, JSON record, trace field, evidence file, or "source
  inventory" without a §5 row changing status (I14, I15).
- An expression compiler, kernel zoo, scalar DSL, or any numerics inside mesh (I7).
- Migrating a caller's operators into mesh (I8), or a model/layer API in mesh.
- A cost model, auto-placement, or a verb inferred from world size (I9).
- A reuse guard, claim check, ready byte, occupancy check, or handshake (I3, I4).
- A default wait: any `waitUntilCompleted`, semaphore, or poll on the publish path (I2).
- A "gold harness", test suite, or benchmark that imports engine internals (I13, I15).
- Deleting a §5 ✓ artifact to rewrite it, unless the rewrite is itself a row.
- Marking a row ✓ on the strength of a build, a deletion, or a document (I13).
- Declaring completion with any row ✗ or ◐.
