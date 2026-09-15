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

## 2. Public surface (closed)

The importable library is the Swift module `Mesh` (`swift/Mesh.swift`) over the C
runtime (`rdma/mesh-call.c`, `mesh-dataflow.c`, `mesh-flow.c`, `mesh-verbs.h`, `mesh.h`):

```
Mesh(region:, rank:, size:, workers:, count:)            // N1 changes `count` to in-flight instances
Mesh.tensor(on:sections:) -> [TensorPart]                 // partial tensor = contiguous sections, each with a rank
Mesh.constant(on:bytes:initialize:) -> TensorPart
TensorFunction.cpu / .metal(device) / .prediction(model) // supplied numerics; input/output view factories
Mesh.call(function, inputs:, outputs:, on:, worker:)
Mesh.map(function, inputs: [[TensorPart]], outputs:, …)
Mesh.send(part, to:, queue:) -> TensorPart
Mesh.broadcast / scatter / allScatter / gather / allGather / allToAll
Mesh.reduce(parts, to:, using:) / reduceScatter(contributions, to:, using:) / allReduce(…)
Mesh.start()                                              // realize storage, bindings, receives, links
Mesh.submit(index)                                        // one value instance through the realized graph
Mesh.syncOnRemoteFill(parts, index:)                      // explicit, diegetic, never default
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

**L3. Publish on visibility, fire on presence.** completion → `mesh_publish(row)` → consumers whose operands are all present are submitted. Ref: Monsoon; Realm; TensorFlow Send/Recv. Check: I2, I3; `mesh_publish` visits only that row's declared uses.

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

**N1. Realize once, submit per instance, unbounded.** `Mesh(…, count:)` → `Mesh(…, inFlight:)`; `submit(index)` accepts any index; instance storage is reclaimed by L7 and reused only after its last reader, with no caller-side guard. Ref: MPI-4 persistent collectives; Pathways; CUDA graphs. Check: `submit(count + 1)` is legal; `start()` prints bytes per in-flight instance; no allocation inside `submit`.

**N2. Early conclusion as a value.** `Mesh.result(index) -> Result<Void, MeshError>`; `MeshError.link(peer, code)`, `.function(call, code)`, `.partialOperand(part)`. Ref: fail-stop; end-to-end argument. Check: I12; the Core ML chain concludes early on a killed peer and the driver re-realizes.

### T — topology and fleet

**T1. Topology is a graph value, observed.** `Topology(nodes:, links: [(a,b): [Link(device, bandwidth, latency)]])`; `Mesh.observe() -> Topology` from live pairings. Ref: JACCL hostfile as the full-mesh case. Check: tree, ring+spurs and full mesh run the same caller with only config changed; no code path assumes `links[(a,b)]` exists.

**T2. Routes belong to placement; mesh forwards on presence.** `Placement(owners:, routes: [(src,dst): [hop…]])`; `send` on a non-adjacent pair is one SEND per hop through canonical intermediate pages. Ref: Patarasuk–Yuan (spanning tree suffices); Pathways; Monsoon. Check: leaf→leaf on a star is two SENDs with no hub wait; per-cut bytes equal declared traffic.

**T3. Multi-link striping per pair.** `send(part, to:, queue:)` where `queue` selects among a pair's links; chunks round-robin when `queue == .all`. Ref: JACCL `ring_impl` wires; TN3205 10 QPs/device. Check: equal bytes per link of a pair in `report`.

**T4. Bounds before runs.** `bounds(program, placement, topology) -> {compute, memory, cut: {cut: s}, path: s, max}`, pure. Ref: `docs/amdahl_superiority.md` four lower bounds; Hockney; LogGP; COSMA. Check: reproduces the ten-minute table (1+3, 2+17, 1+1); every report prints `measured / bounds.max`.

**T5. Retopology is a `Result`; the driver re-realizes.** `MeshError.topology(lost:, gained:)`; driver: `observe(); Mesh(topology, placement.restrict(topology)).start()`. Ref: fail-stop; Dean–Ghemawat; `RDMA-RULES.md`. Check: pull one cable mid-run on a ring: early conclusion names the link; re-realized program completes on the spanning tree; re-plug is visible to the next `observe()`.

**T6. Replicas are placement, not protocol.** `Placement(…, replicas: [section: [rank, rank]])`; readers take the first present. Ref: Legion physical instances; end-to-end. Check: no ack/retry/health code; losing a link changes `bounds()` not correctness.

**T7. Multi-tenant fabric by lease.** bridge serves N clients; `Mesh(…, lease: Lease(qpsPerLink:, arenaBytes:))`; setup returns errno when the lease exceeds `max_qp`/arena. Ref: NCCL communicator per job; PMIx. Check: two users' programs run on disjoint QPs/arena; `kill -TERM` of one leaves the other untouched; no `SIGKILL` path.

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
| 2 | S2 TX/RX threads per link | ✓ (unrun on RDMA) | `c9d9e18` |
| 3 | S3 chunked transport invisible | ✓ (unrun) | `98742c8` |
| 4 | S4 bounded pairing | ✓ | `06dcdb3`; checked nonblocking sockets, one deadline across all pairing exchanges |
| 5 | L1 partial tensor | ✓ | `swift/Mesh.swift` `TensorPart` |
| 6 | L2 supplied function | ✓ | `TensorFunction.cpu/.metal/.prediction` |
| 7 | L3 publish/fire | ✓ (unrun) | `mesh-call.c` `mesh_publish` |
| 8 | L4 ten verbs | ✓ | `Mesh.swift:279-365` |
| 9 | L6 any combine | ✓ | by construction |
| 10 | L7 lifetime | ✓ (unrun) | `mesh_buffer_retain`/`mesh_collect` |
| 11 | L8 explicit sync + counterexample | ✓ | `examples/sync-on-remote-fill.swift` |
| 12 | P2 push both repos | ✗ | — |
| 13 | P1 importable; Core ML chain run on the pair at ABI 43 | ◐ | bridges still at the pre-rewrite ABI; `links=()` |
| 14 | L5 Partial type | ✗ | — |
| 15 | N1 unbounded instances | ◐ | `count` finite |
| 16 | N2 Result surface | ◐ | — |
| 17 | E1 engine layer via Mesh | ✗ | — |
| 18 | E2 public measurement + Karp–Flatt | ✗ | — |
| 19 | E3 depthwise chain not slower | ✗ | — |
| 20 | G1 contraction/Gram as the same calls | ◐ (FFN shown, Gram not) | `examples/coreml-chain.swift` |
| 21 | G2 indices as data (caller pattern) | ✗ | — |
| 22 | G3 two callers | ✗ | — |
| 23 | T1 topology observed | ◐ | links only |
| 24 | T3 multi-link striping | ✗ | — |
| 25 | T2 routes/forwarding | ✗ | — |
| 26 | T4 bounds() | ✗ | — |
| 27 | T5 retopology Result | ✗ | — |
| 28 | T6 replicas | ✗ | — |
| 29 | T7 lease / multi-tenant | ✗ | — |

Rows 20–29 are required for the 4× M5 Ultra + 4× M4 Pro deployment and for the solver
caller; they are not optional and not "later" — they are after row 19. Rows whose files
are disjoint may be worked in parallel worktrees: {14,15,16} share `Mesh.swift`/`mesh-call.c`;
{23,24} share `mesh-flow.c`; {17},{18},{26} are independent; {13},{19} need the link
and run one at a time (`RDMA-RULES.md`: one experiment at a time).

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
