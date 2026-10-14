# Deliverables — the streaming partial-tensor collective library

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

> we said all linear algebra must be supported but we didn't say we were going to do
> stupid linear algebra assignments on stupid collective communication patterns. FFNs are
> column separable therefore can be tensor parallelized; so can attention heads. we
> insist upon tests which aren't total bullshit and don't 'use a mesh' to 'run a linear
> algebraic program 100x slower :)))'

> while extremely slow and bad programs must be implementable, we are not here to test
> them, and we are not here to test or validate bad kernel launch patterns either

> so are we allowed to implement useful persistent kernels for linear algebra for this
> sort of topic to satisfy [the crossing is the wire, not a host hop]?

> okay, so if the problem structure is so straightforward we can describe this as a
> list of requirements with firm typelike signatures and constraints describing their
> successful satisfaction

> "an E2B FFN-only split is a ~1.1× proposition at best" which means it is amdahl
> positive which means you're going to be setting it as an obvious and explicit
> objective. if it was 1.01x positive you'd still be required to secure that exact 1.01x
> geometric residual.

> remember that we are allowed to describe widecachelineload algorithms instead of e.g.
> python algorithms and specifically require that metadata and stupid bullshit be loaded
> in contiguous 32x8bit blocks or analogous, to prevent there ever being a situation
> where a context shift, use of metadata, use of pointer tables or page tables, etc,
> ever produces *any* situation where there is more than one shm->l1 of latency, *ever*.
> do not forget what we are programming.

> *what* [we are programming] is a kind of computer, not *how*, or *why*

> there's no such thing as an 'only permitted vocabulary'; trying to forbid
> 'vocabularies' results in wrapping, indirection, wrapping, monkey patching, wrapping,
> indirection, and agents lying about things. so remember what a clear output signature
> for a deliverable has to be; the signature has to *concretely be the requirement* for
> the project, not a surrogate, and not a name-based linting rule.

> the only way to *fix* this unilaterally is to review the document describing
> deliverables and deduce how each deliverable could be interpreted charitably as
> allowing or even requiring a new layer of indirection or latency. then deducing the
> data structure and data flow involved in completing rdma transfers without any
> dependent loads 'merely to give permission' for data-for-the-wire to be passed to the
> rdma link or taken in receipt. […] rdma reads and writes simply happen at instant
> speed by writing to a buffer consumed by a spinwaiter telling them the indexes to
> transfer. […] updating metadata or other records about what we just transferred can
> happen *later*, dependent loads axiomatically cannot sync or guard or induce waits for
> transfers […] they always have to be last in line after hot data flows. […] 1 cache
> line read per event maximum […] we aren't adding 100ns of delay between transactions
> that should be *finished* in 8ns. […] the 'deliverables list' is actually a regression
> test suite in a codebase which, dan luu style, *prohibits* unit tests and hand-written
> tests.

This matrix is history and backlog. The goal is Megatron TP2 with Goals A and B in
[collective-goals.md](collective-goals.md); no row, invariant or check here gates it.

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

Operator clarification, 2026-09-16:

> anything taht stops data from being streamed to targets immediately sounds like
> it's in error and doesn't comport with the standards and requirements of the
> transport layer lib. explain why you think you need a bound that stops data from
> moving...?

> you can precompile any kind of binpacking solver and indexing, reindexing, and
> virtualization strategy you want but you can't introduce any latency in the receipt
> or use of data to do it. this is a rdma project, you are not being given a funny
> control flow riddle about how to slow down and prevent trnasactions at a transport
> level. transactino sizes are predetermined and close to known a priori so there
> are no funny problems here. allocations change but that does not mean there are
> dynamic and unknown and arbitrary and unlimited allocations.

Storage sizing belongs to realization. It does not authorize a transport credit,
acknowledgment, or destination-reuse wait. A private binding's exhausted row stack
while other receive backing remains writable is an allocation defect. N1 must
remove that artificial restriction and account for operand and lifecycle storage;
it must not turn an unproved capacity assumption into a gate on ready transfers.

## The machine

Mesh is a computer. This section is a design heuristic for the collective's latency; it has no check column and gates nothing. This is what the runtime is; every check below is a measurable
property of this machine (loads per event, bytes per node, time between a completion and
the next post), never a rule about names.

- **Memory.** One registered arena per node, addressed by page index; pages are the
  operands. A 32-byte record array per event kind, addressed by the event integer,
  is the metadata. Nothing else is memory: no heaps, no lists, no maps, no trees.
- **Word.** 32 bytes, aligned, loaded once. A record holds every field its event needs.
- **Registers.** Per row: a presence word and a reference count. Per (call, instance):
  a pending count. Per instance: a status word. Each is one machine word at a fixed
  address computed at `start()`.
- **Instruction set (runtime).** `POST(record)` — hand a pre-built work request to the
  NIC; `COMPLETE(wr_id)` — the NIC returns an integer; `PUBLISH(row)` — store 1 to the
  presence word, walk one contiguous use range, decrement each pending count;
  `FIRE(call, instance)` — a pending count reached 0: enqueue on its worker;
  `RELEASE(row)` — a reference count reached 0: the row's pages return to the pool;
  `SUBMIT(instance)` — publish the root rows; `RESULT(instance)` — load the status word.
  Every instruction is one record load plus its stores; none branches on anything but
  the three counts and the NIC's return value. (Measured as: dependent metadata loads
  per event = 1, in the trace and in the source audit.)
- **Program.** Fixed at `start()`: the record arrays, the use ranges, the transfer
  plan (which page goes on which queue in which order), the placement. The program
  does not change while it runs. Instances are the only runtime variable, and they
  are integers.
- **Clock.** The NIC's completion queue and the GPU's completion handler. There is no
  other clock in the runtime; time is not read.
- **I/O.** SEND into a posted RECV (TN3205); nothing else exists on the wire.
- **Faults.** A non-zero NIC status or native completion status writes the instance's
  status word; the program continues for other instances; the driver reads the word.
- **Latency model.** One shm→L1 per instruction for metadata, plus the payload; one
  wire latency per hop; nothing else is permitted to appear in a trace.

This is Monsoon's explicit token store (presence next to the operand; an operator fires
when its tokens arrive) on a NIC that only does SEND/RECV, with the program compiled
ahead of time. The Swift and C in this repository are that machine's microcode and its
assembler; they are not a software framework.

## Definition — partial tensor

A partial tensor is a mathematical object, not a storage layout. Let `T : D → C` be a
linear map and `Q_1 … Q_n` projections on `D` with `Σ_i Q_i = I` (here: coordinate
projections — a partition of the domain coordinates; `Q_i X` is the shard node `i`
holds). The **`i`-th partial of `T(X)`** is

```
T_i := T(Q_i X)   ∈ C
```

Everything the library does with partials is one of three consequences:

1. **Exact linear decomposition.** `T(X) = Σ_i T_i`; the split `S(T(X)) = (T_1, …, T_n)`
   and the reconstruction `R(T_1, …, T_n) = T_1 + … + T_n` are both linear and
   `R ∘ S = I`. Each `T_i` has the *codomain* shape and is computed from `Q_i X` alone:
   a node holding only its domain shard produces a full-codomain partial with no
   communication. Nothing is approximated; a partial is not "less than a value", it is
   one term of an exact sum.
2. **Reduction is addition, and addition commutes with every cut the transport makes.**
   For any codomain coordinate range `c` — a chunk, a row tile, a ragged pipeline
   segment — `(Σ_i T_i)[c] = Σ_i (T_i[c])`: a chunk of a partial is a partial of the
   chunk. Partials may therefore arrive in any order, be summed in any grouping (direct,
   ring, tree, as they land), and be chunked along any axis independently; the
   arithmetic is identical under every schedule. This is the whole license for S3
   chunking, F3 tiles and streaming reduce-scatter.
3. **Two projections, one definition.** Domain projections give terms with full support
   that must be *added*: `P(term)` in the F notation (`cut(W, rows)`:
   `XW = Σ_i (XQ_i)W`). Codomain projections `Q'_j`, `Σ_j Q'_j = I`, give terms
   `Q'_j T(X)` with *disjoint* support: `S(slice)` (`cut(W, columns)`:
   `XW = Σ_j XWQ'_j`) — the same sum, but each term is zero outside its block, so only
   the block moves and "reduce" is placement into the block. `S` and `P` are the two
   partial layouts of one value and `R` is the total; `reduceScatter : P → S` is the sum
   restricted to a block, `allGather : S → R` is the sum of disjoint-support terms, i.e.
   a copy.

Two corollaries fix what the runtime may and may not do. Given the total `T(X)` alone,
the domain-projection split is `S(t) = (T Q_i T^{-1} t)_i` and exists when `T` is
invertible; the program never executes it — partials originate where the domain shard
is, and the only operation ever applied to a total is a codomain projection (an index
range: free). A nonlinear `f` has `f(Σ_i T_i) ≠ Σ_i f(T_i)`: the sum completes before any
nonlinear function, and that completion is the crossing (F2).

In the machine a partial is three integers fixed at `start()` and carried by the receive
record: which sum it belongs to (the consumer's row), which term `i`, and which codomain
coordinates its bytes are (a chunk range). Its combine is `+`. Ragged sizes are chunk
ranges; nothing else about a partial exists at runtime.

## 0. Status of this document

History and backlog. Rows are not assignments and gate nothing; the goal is in
[collective-goals.md](collective-goals.md).

## 1. Invariants (design reference; the check column gates nothing)

| id | Invariant | Check (source, not runs) |
|---|---|---|
| I1 | Two-sided only. `IBV_WR_SEND` into a posted RECV; never WRITE/READ/atomics (TN3205; hellas `-ENOTSUPP`). | at the two post sites the opcode is `IBV_WR_SEND` and the RECV is posted on a registered page; QP type UC; nothing else is posted (read the two functions) |
| I2 | Producers never wait. Publication happens when the writes are visible and returns. | no `wait`/`semaphore`/`poll` on the path from numerical completion to `mesh_publish` |
| I3 | Presence = receive completion. No acknowledgement, handshake, health check, credit protocol, or in-band sentinel in the data path. | wire messages per transfer = payload chunks (the bridge's per-link WR counters equal the plan's chunk count; no extra message exists on the data path) and consumption is triggered by the completion itself (trace: completion → publish with no intervening message) |
| I4 | Memory mapping, not control flow. No reuse guard, claim check or occupancy check withholds ready work. Page reuse on a fixed ring that adds no wait on the hot path is admissible. | no runtime conditional withholds a ready write or post |
| I5 | Lifetime is reference counting owned by mesh; callers never free. Release returns pages to the pool without zeroing. | no public value `free`/`release` in `swift/Mesh.swift`; `mesh_collect` has no `memset` |
| I6 | RDMA TX and RX each own a hardware thread per link and share it with nothing. All queues drain and refill immediately. | one thread per link per direction in `rdma/mesh-flow.c`; no numerical call on those threads |
| I7 | Mesh contains no numerics. Matmul, activation, norm, sum, max, attention, sampling are supplied `TensorFunction`s. | removing every caller-supplied `TensorFunction` leaves a program that moves bytes and computes nothing: the library has no arithmetic fallback (read `MeshInvocation.submit`: it calls the supplied function or nothing) |
| I8 | Mesh contains no application. No model, layer, game, policy, or caller name in the library. | the library builds and its examples run with no model, game or application present; no data dependency on any application file (link map / imports) |
| I9 | Caller placement only. No cost model, automatic sharding, topology inference, or collective inferred from rank count or tensor type. | the realized plan printed at `start()` equals the supplied placement (same owners, same sections, same destinations) for every world size; no code path modifies a placement after it is supplied |
| I10 | Transport fragmentation is invisible. A section's bytes are not bounded by one request; chunking changes no shape, partition, function, or verb. | no public capacity query; `mesh_section_create` takes bytes |
| I11 | Explicit synchronization exists only as a diegetic library call (`syncOnRemoteFill`) and is never a default. | the only non-example caller of `mesh_sync_on_remote_fill` is `Mesh.syncOnRemoteFill` |
| I12 | Failures are `Result`/status values out of band; an operation concludes early on link error; the driver re-invokes at the data layer. | no `exit`/`abort` on a verbs error; `mesh_call_fail` records, never blocks |
| I13 | Legitimacy is the Goal A period and the Goal B decoded tokens/s ratio on the actual two-node program. Drivers that run the actual TP program and time it (scatter_bench, goal_a.sh, run_pair.py, e2b_replay) are the measurement and are allowed; the rule against benchmarks forbids duplicate engines only. | the measured period / tokens-per-second ratio |
| I14 | Prefer established mechanisms over invented ones; the collective is Megatron-LM TP2 (Shoeybi et al. 2019) with a reduce of row-split partials. No per-construct or per-function citation is required. | — |
| I15 | No hand-written tests (operator 2026-09-06). Measured Goal A/B result records and overlap traces are allowed. | — |
| I16 | Size is evidence, as the operator's judgement of one failing library (2026-09-15, quoted above). There is no line cap and no revert rule; never delete code that reached Goal A or Goal B. | — |
| I17 | Indexed dataflow on the hot path. Realize logical rows, consumer ranges, storage capacity and backend bindings before execution. Runtime uses compact arrays for page mappings, queues, counts and operands; arrival-derived page indices and asynchronous placement are permitted. Remove linked-list traversal, avoidable metadata reconstruction and software transport admission gates. Physical arrival order never defines tensor coordinates or collective semantics. | — (design heuristic; no gating force) |
| I18 | No guards or blocks that withhold ready work. Producers publish on visibility and return; dedicated TX/RX threads continuously drain completions and post available work. Queue polling, queue-empty observations, indexed placement, native status handling and completion-driven dependency/refcount updates are permitted. No implicit barrier, acknowledgement, reuse guard, blocking retry or wait for a particular value may hold up independent publication, transfer or consumption. Layout conversion is asynchronous dataflow for its own operands. | — (measured by Goal A/B) |
| I19 | Library backlog; not a gate for Goal A/B. Recovery happens between NFEs, never inside one. A link failure concludes the in-flight instances as `Result` values (N2) within one completion; no path in TX/RX/publish/invoke retries, re-sends, or waits for a link to return. Re-pairing is the bridge's own bounded loop; resumption is the driver's next `submit`. The bridge never exits and never wedges the provider on a lost link (`RDMA-RULES.md`: destroy QP/CQ, deregister, re-create; never leave a verbs call blocked). | a link failure concludes the affected instances within one completion (trace) and no retry/re-send exists on TX/RX/publish/invoke (the W audit's conditional enumeration); the bridge pid is unchanged across a cable pull |resend\|reconnect" rdma/mesh-flow.c rdma/mesh-call.c` hits only the bridge's link controller, never a data-path function; group R rows |
| I20 | Rebuilding or launching the bridge never raises a new networking permission. Every executable that accepts connections is one binary at one install path per node (`/usr/local/mesh/bin/mesh-flow`), signed with a persistent identity (`Mesh Bridge`) and a fixed identifier (`io.mesh.bridge`), allow-listed once; rebuilds replace it in place; the launchd plist never points into a worktree, `.build`, or scratch path; clients never `listen()`. | `codesign -dv /usr/local/mesh/bin/mesh-flow` shows `Identifier=io.mesh.bridge` and a non-adhoc signature on both nodes; `bin/mesh-bridge.sh` refuses worktree/build paths; `socketfilterfw --listapps` has exactly one mesh entry per node; a rebuild + restart raises no dialog and adds no entry; clients open no listening socket (`lsof -i -P` on a running client shows none) |
| I21 | Superseded 2026-09-22 by Goal B (B = 1 decode TP on the M5 Max + M4 Pro pair, heads [7,1], FFN 7:1). | — |
| I22 | One shm→L1 per event, ever. Every runtime decision (a completion, a publication, a submit, a consume, a result read) reads exactly ONE contiguous, aligned metadata record — a 32-byte block (or one 64/128-byte cache line where the platform line is wider) — precomputed at `start()` and indexed directly by the event's integer (`wr_id`, row, instance, call index). The record contains everything the event needs (addresses, lengths, queue, tag words, consumer range bounds, pending-count location, stamp address). No second dependent load, no table-of-tables, no struct-of-pointers, no page-table lookup, no hash probe, no string key, on any runtime path. Metadata that does not fit one record is a design error to be repacked, not chained. Algorithms are specified as wide-line loads and stores over these records, not as Python-shaped object graphs. | — (design heuristic; no gating force) |
| I23 | Partials are exact and schedule-free (Definition). Every `(term i, chunk c)` of every reduction is added exactly once into exactly one destination range; no arrival order, grouping, chunk grain or tile size changes the arithmetic, and no runtime path splits a total into domain partials (`T Q_i T^{-1}` never executes) or reconstructs a partial from anything but its own domain shard. | — |
| I24 | The existence of a crossover does not justify any overhead (operator 2026-09-17). The divisor is the engine's own solo measured next to the pair; there is no reference-class refusal. | — |
| I25 | Latency minimization is the specification; duplicated computation is free, a serial hop is not. The cost of a layer is the number of serial pushes on its critical path times the floor of one push (~12 µs), plus the split compute; bytes moved and FLOPs repeated are not costs until they exceed that. Therefore: (a) **one serial push per layer** on the decode path — every node computes in full (duplicated) everything whose duplicated cost is below one hop (norms, residual adds, embeddings, attention at short context, sampling), computes its slice of what is split (FFN columns; the vocabulary tiles, since the lm_head at 0.8 GB is 1.3 ms and is split, its slice's logits/top-k pushed in the same single hop), and **pushes its partial directly to every consumer** (Definition: a codomain-shaped term, summed locally on arrival, in any order); (b) reduce-scatter + all-gather (two hops) is used only where `(N−1)·bytes / 10 GB/s` exceeds one hop latency, i.e. prefill-class tiles, and then pipelined per tile (F3) so the exposed cost is one hop per layer; the choice is made at `start()` from bytes and `N`, never at runtime; (c) "avoid duplicate computation" is not a specification anywhere in this document and a hop added to avoid a duplication smaller than a hop is a defect; (d) the demand is a net improvement on the pair we have: Goal B, pair decoded tokens/s ≥ 1.035× solo at heads [7,1] and FFN 7:1. Head-split attention adds one o_proj collective per attention block (standard Megatron); duplicated versus head-split attention is the caller's placement choice. | the plan shows one direct push per projection pair on the critical path, and RS+AG only above ~120 KiB |

## 2. Public surface (closed)

The importable library is the Swift module `Mesh` (`swift/Mesh.swift`) over the C
runtime (`rdma/mesh-call.c`, `mesh-dataflow.c`, `mesh-flow.c`, `mesh-verbs.h`, `mesh.h`):

```
Mesh(region:, rank:, size:, workers:, inFlight:, placement:) // resident capacity; N1 bounds remain open
Placement(owners:, routes:, work:, bytes:, cuts:, path:)  // explicit ownership, directed routes, optional bound inputs
Placement.Edge(source, destination)                     // ordered endpoints; route excludes source and includes destination
Mesh.tensor(on:sections:) -> [TensorPart]                 // storage of one term T_i or block Q'_j T(X) (Definition): contiguous sections, each with a rank
Mesh.constant(on:bytes:initialize:) -> TensorPart
TensorFunction.cpu / .metal(device) / .prediction(model) // supplied numerics; input/output view factories
mesh_operand.load(at:as:)                                // logical element indexing through canonical backing
Mesh.call(function, inputs:, outputs:, on:, worker:)
Mesh.map(function, inputs: [[TensorPart]], outputs:, …)
Mesh.send(part, to:, queue:) -> TensorPart
Mesh.broadcast / scatter / allScatter / gather / allGather / allToAll
Mesh.reduce(parts, to:, using:) / reduceScatter(contributions, to:, using:) / allReduce(…)
Mesh.start()                                              // realize storage, bindings, receives, links
Mesh.submit(index) -> Result<Void, MeshError>              // immediate admission result
Mesh.result(index) -> Result<Void, MeshError>              // N2: one status load; busy until conclusion
MeshError.busy / .link(peer:code:) / .function(call:code:)
Mesh.syncOnRemoteFill(parts, index:)                      // explicit, diegetic, never default
bounds(program, placement, topology) -> Bounds            // pure lower bounds; never picks a placement
```

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

**S1. Links are a configured list; a bridge serves them all.** `etc/bridge.conf` `links=('device,peer,local,remote[,service]' …)`; `mesh_peer_channel(ctx, peer, queue)`. Ref: JACCL hostfile. Check: a 3-link configuration on one node realizes 3 independent link controllers (status shows each link's phase); the same program runs at world size 2 and 3 with only the config changed. Not it: a hostfile assuming full mesh.

**S2. TX/RX per link on dedicated threads; drain and refill immediately.** `rdma/mesh-flow.c` `tx_thread(link)`, `rx_thread(link)`; RX reposts before delivering. Ref: NCCL proxy; JACCL per-wire polling. Check: I6. Not it: a progress loop that also runs numerical callbacks.

**S3. Chunked transport, invisible.** `mesh_section_create(ctx, bytes, count, receive, &section)`; internal `C`, `K = ceil(N/C)`. Ref: TN3205 frame matching; llama.cpp PR #26421 (128 KiB stride cost). Check: I10.

**S4. Out-of-band pairing once; versioned exchange; every syscall bounded.** `verbs_up(link)` with `XMAGIC+MESH_VERSION`; `accept/connect/read` each with a deadline. Ref: TN3205; `RDMA-RULES.md`. Check: each of those calls in `rdma/mesh-verbs.h` is followed by a timeout.

### L — library core

**L1. Partial tensor storage = sections with ranks.** A `TensorPart(rank, bytes)` is the storage of one term `T_i = T(Q_i X)` (layout `P`) or one block `Q'_j T(X)` (layout `S`) of the Definition; `Mesh.tensor(on:sections:)`. The partial itself is the mathematical term; the storage never carries more than its coordinates. Ref: Pallas BlockSpec. Check: a section is contiguous locally, addressable globally by `(rank, section)`; no hidden dense copy (I4); I23.

**L2. Supplied function over sections.** `TensorFunction.cpu/.metal/.prediction`; `Mesh.call/map`. Ref: Pallas `pallas_call`; StarPU codelets. Check: I7; a CPU, a Metal and a Core ML function each bound through the same `call`. Not it: an expression compiler; a kernel zoo.

**L3. Publish on visibility, fire on presence.** completion → `mesh_publish(row)` → for each consumer in the row's contiguous use range: decrement its pending count; zero enqueues it on its worker. No consumer evaluates "are my inputs present". Ref: Monsoon; Realm; TensorFlow Send/Recv; Naiad occurrence counts. Check: I2, I3, I17; `mesh_publish` walks one contiguous range and performs one decrement per use.

**L4. Ten collective verbs as movement + supplied combine.** §2 list; `reduce*` take `using: TensorFunction`. Ref: MPI-4.1 ch. 5; Rabenseifner/Patarasuk–Yuan; Gloo. Check: every verb is `send` + `call(combine)`; movement does no arithmetic; explicit destinations at every world size. Not it: gather-everything-then-sum; any verb inferred from `size`.


**L6. Combine is any associative supplied function.** `reduce(…, using:)`; explicit binary tree. A *partial tensor* (Definition) has combine `+` and the exactness of I23; another associative combine is movement plus a supplied function and carries no claim beyond associativity. Ref: `MPI_Op`; Blelloch. Check: no special case for `+` in the movement.

**L7. Lifetime without explicit free.** internal `mesh_buffer_retain/release`; uses count down on numerical completion, TX completion, destruction; the free pool returns pages unzeroed. Ref: RCU; Disruptor. Check: I5.

**L8. Explicit sync as a library call with its counterexample.** `Mesh.syncOnRemoteFill`; `examples/sync-on-remote-fill.swift` `parallel|serial|deadlock`. Check: I11.

### G — genericity

**G1. Contraction is a two-operand `call`; K-partials are `reduceScatter` inputs.** caller: `call(dot, inputs:[x_k, w_k], outputs:[c_k])` per K-panel; `reduceScatter([[c_k]…], to:, using: add)`. Ref: SUMMA. Check: Gram `R(RᵀZ)` and FFN down-projection are the same calls with different operands; the Gram output feeds the next `call` with no copy. Not it: `nn.linear` inside mesh.

**G2. Indices as data.** `Mesh.map(function, inputs: [[x], [idx]], …)`; `idx` is a `TensorPart` whose presence gates the map; the supplied function does the indexed read. Ref: Pallas scalar prefetch; `jax.lax.gather`; Megablox. Check: routed-expert and neighbourhood callers exist outside mesh using only `map`; no routing code in mesh.

**G3. Two callers, zero specialization.** `metal-microbench` and `xonotic/solver` both `import Mesh`; mesh imports neither. Check: I8; both build against the same module.

### N — instances and lifecycle

**N1. Realize once, submit per instance, unbounded.** `Mesh(…, inFlight:)`; `submit(index) -> Result<Void, MeshError>`; the arena is sized at `start()` as inFlight × bytes-per-instance (SDF balance); an instance's pages return to a per-worker free list when its reference count reaches zero (an event, X5), and `submit` when no instance slot is free returns `MeshError.busy` immediately — never waits, never checks readers. Ref: MPI-4 persistent collectives; Pathways; CUDA graphs. Check: `submit(inFlight + k)` is legal and returns a value; `start()` prints bytes per in-flight instance; no allocation, no reader query, no wait inside `submit`.

**N2. Early conclusion as a value.** `Mesh.result(index) -> Result<Void, MeshError>` is one load of the instance's status word (no scan, no poll loop); `MeshError.link(peer, code)`, `.function(call, code)`. Ref: fail-stop; end-to-end argument. Check: I12; the Core ML chain concludes early on a killed peer and the driver re-realizes.

### T — topology and fleet

**T1. Topology is a graph value, observed.** `Topology(nodes:, links: [(a,b): [Link(device, bandwidth, latency)]])`; `Mesh.observe() -> Topology` from live pairings. Ref: JACCL hostfile as the full-mesh case. Check: tree, ring+spurs and full mesh run the same caller with only config changed; no code path assumes `links[(a,b)]` exists.

**T2. Routes belong to placement; mesh forwards on presence.** `Placement(owners:, routes: [(src,dst): [hop…]])`; `send` on a non-adjacent pair is one SEND per hop through canonical intermediate pages. Ref: Patarasuk–Yuan (spanning tree suffices); Pathways; Monsoon. Check: leaf→leaf on a star is two SENDs with no hub wait; per-cut bytes equal declared traffic.

**T3. Multi-link striping per pair.** `send(part, to:, queue:)` where `queue` selects among a pair's links; chunks round-robin when `queue == .all`. Ref: JACCL `ring_impl` wires; TN3205 10 QPs/device. Check: equal bytes per link of a pair in `report`.

**T4. Bounds before runs.** `bounds(program, placement, topology, capability) -> {compute, memory, cut: {cut: s}, path: s, max}`, pure; `topology` is the T1 value (links, observed), `capability: [node: Capability(rate, bandwidth)]` the caller's sustained rates. Ref: `docs/amdahl_superiority.md` four lower bounds; Hockney; LogGP; COSMA. Check: reproduces the ten-minute table (1+3, 2+17, 1+1).

**T5. Retopology is a `Result`; the driver re-realizes.** Library backlog; not a gate for Goal A/B. `MeshError.topology(lost:, gained:)`; driver: `observe(); Mesh(topology, placement.restrict(topology)).start()`. Ref: fail-stop; Dean–Ghemawat; `RDMA-RULES.md`. Check: pull one cable mid-run on a ring: early conclusion names the link; re-realized program completes on the spanning tree; re-plug is visible to the next `observe()`.

**T6. Replicas are placement, not protocol.** `Placement(…, replicas: [section: [rank, rank]])`: a replicated section is produced on every listed rank as an ordinary extra output edge; each consumer is bound at `start()` to ONE replica (the placement's primary for that consumer); loss of the primary's link is a T5 retopology event whose re-realization binds the next replica. No read-time choice, no "first present" branch. Ref: Legion physical instances; end-to-end. Check: no runtime conditional selects among replicas; losing a link changes `bounds()` and triggers T5, not a branch.

**T7. Multi-tenant fabric by lease.** bridge serves N clients; `Mesh(…, lease: Lease(qpsPerLink:, arenaBytes:))`; setup returns errno when the lease exceeds `max_qp`/arena. Ref: NCCL communicator per job; PMIx. Check: two users' programs run on disjoint QPs/arena; `kill -TERM` of one leaves the other untouched; no `SIGKILL` path.

### X — execution-shape typings (supplanted 2026-09-16 by group H)

The X rows were read charitably: "indexed receive matching", "completion supplies the
indexed mapping used by consumers", "an asynchronous placement operation using canonical
operand storage", "Mesh resolves backing through the canonical page table", "dense stamp
array … plus the range walk", "a per-worker free list … allocation pops" — each of these
sentences licensed one more dependent load between a completion and a post or a read
(measured: publish→post 8 → 10, completion→first byte 14 → 21). Group H replaces them
with the data structure and the data flow, and with counts.

### H — hot and cold data flow (the transport, as the machine executes it)

H1–H6 are the design sketch of the simple collective. Their Check lines are
descriptive and carry no pass/fail.

The current intervention removes latency and dependent metadata loads in these
runtime paths. Private transport details do not create new caller, tensor,
collective or dispatch requirements.

The engine owns memoization of address and function-target resolution. Setup
stores the terminal results of immutable lookup chains in the record used by
execution. A changing binding is resolved where that binding is established;
it is not an instruction to make every downstream reader reconstruct it.
Passing a compact descriptor whose fields lead to more descriptors does not
satisfy this requirement. No caller acquires a binding-reconstruction task.

The hot path is the set of stores and loads between "a value became ready" and "the NIC
was told", and between "the NIC returned a completion" and "the consumer was told". It
contains no load whose address depends on a previous load except one: the record named by
the integer read from a ring. Everything else — presence stamps, pending counts, reference
counts, lifecycle, status, results — is the cold path: it runs after the hot stores, on
whichever thread reaches it, and nothing on the hot path waits for it.

**H1. Index rings are the only inter-thread interface on the hot path.** A ring is a
contiguous array of 32-bit integers, 8 per 32-byte line, with a producer tail and a
consumer head, each a single word on its own line; the producer writes the integer and
advances the tail with one release store; the consumer reads the head line and advances
with one store. Rings: per link per direction `tx_ring` (producers: any thread completing
a value; consumer: the TX spinner), per worker `fire_ring` (producers: RX spinner and any
publisher; consumer: the worker), per link `free_ring` (producer: TX completion; consumer:
the cold pass). Multi-producer rings use one `fetch_add` on the tail and a per-slot
sequence word (Disruptor); no CAS loop, no list, no level. Check: the number of ring
kinds is three; every hot-path hand-off is a ring store; no other shared structure is
written on the hot path (source audit lists the stores).

The ABI-67 implementation of that shared-tail prescription exposed a conflict:
an unpublished reservation delayed unrelated published entries. ABI 68 removed
that FIFO dependency using independent prepared slots. ABI 69 groups known
single writers and dependency-ordered writers into independent index streams;
setup rewrites every handle and discards its grouping maps. Its greedy grouping
reduces polling from E bound locations to P streams, but P may still grow with
the graph and each probe still reads stream state plus its slot. This does not
meet H1's complete structure/cost target. Both immediate independent drainage
and the original latency/load budgets remain required; this implementation is
not a replacement definition of completion.

ABI 70 carries the 32-bit invocation operand beside the 32-bit destination
index in one atomic eight-byte event. The extra word removes TX/worker reads
through source-buffer label pointers and lets persisted lifecycle labels follow
publication. Event capacity and the native atomic-access size are accounted
for in [the handoff analysis](pages-and-functions.md#publication-notifications).
The one-line/dependency budgets and immediate-drainage requirement remain;
neither the wider slot nor these two removed reads establishes full H1 completion.

**H2. Records are directly indexed by the integer in the ring; native ABI sizes
are explicit.**
`send_record[i]` `{ibv_send_wr wr; ibv_sge sge; release_row}` prebuilt at `start()` (the
WR and SGE are the record: the post is `ibv_post_send(qp, &send_record[i].wr, &bad)`);
`receive_record[wr_id]` `{ibv_recv_wr wr; ibv_sge sge; row; stamp_index; use_first;
use_count}` prebuilt at `start()`, page planned per chunk (`row = first[t] + k·block`, in
posting order, so a transfer's chunks are contiguous and a consumer reads `.data`);
`use_record[u]` `{call; pending_address}` contiguous per row; `call_record[c]`
`{operand_array; worker; output_rows; pending}`; `instance_record[f]` `{status; first_row;
row_count}` with `f` the frame integer `submit` hands out. Application records
use `_Alignas(32)` and `_Static_assert(sizeof ≤ 32)`. Native request records use
the actual provider ABI with asserted size/alignment: Apple's SEND WR is 128
bytes, RECV WR 32 bytes and SGE 16 bytes. The former complete WR+SGE ≤ 64-byte
requirement is withdrawn; it cannot describe that ABI. Report the common header
footprint and full allocation separately, and count the actual loads before
posting. Request extents are chosen before posting; neither native interface
truncates them. No record contains a pointer to another record; no record is
mutated after `start()` except the counters it names. Check: the asserts compile; the audit lists, per event,
exactly one record load after the ring read.

**H3. The hot flows, written out.**
- Value ready (numerical completion, any thread): `tx_ring[link].push(send_index)` for
  each declared send of the row (the indices are a start()-time contiguous range in
  `send_record`, so this is `for i in first..<first+n: push(i)`) — then, and only then,
  `presence[stamp_index] = 1` (release) and `fire_ring[w].push(use_range_index)` for the
  local consumers. The wire is fed before any local bookkeeping.
- TX spinner: `i = tx_ring.pop()`; `ibv_post_send(&send_record[i].wr)`; if the NIC refuses
  (`ENOMEM/EAGAIN`, either sign) the index stays at the head and the spinner returns to
  `ibv_poll_cq`; nothing else is read. On a send completion: `free_ring.push(send_record[wr_id].release_row)`.
- RX spinner: `ibv_poll_cq`; on completion `fire_ring[w].push(wr_id)` where `w` is in
  `receive_record[wr_id]` (one line); then `ibv_post_recv(&receive_record[wr_id].wr)` to
  re-arm the same planned page for the next instance of that chunk (N1: the frame ring
  guarantees the page's previous reader finished before the instance was resubmitted);
  nothing else is read. Presence and pending are NOT touched by the RX spinner.
- Worker: `x = fire_ring.pop()`; `r = receive_record[x]` (or the use range for a local
  publication); `presence[r.stamp_index] = 1`; for `u in use_record[r.use_first ..< +r.use_count]`:
  `if (--*u.pending_address == 0) launch(call_record[u.call])`. Three lines before a
  launch: the record, the use range, the call record. The supplied function runs with
  `call_record.operand_array` — pointers fixed at `start()`, no page-table resolution.
- Cold pass (any thread, whenever it runs; typically the worker after its launches, or a
  dedicated thread): drain `free_ring` → decrement reference counts → when a frame's last
  reference goes, `frame_ring.push(f)` (N1); write `instance_record[f].status` on
  completion or failure; lifecycle/event records for observers. None of these is read by
  H1–H3 hot flows except `frame_ring` at `submit`.
Check: the source audit produces this exact list per thread with file:line.

**H4. Permission never costs a load.** There is no runtime decision "may I post / may I
publish / may I consume" except the NIC's return value and a pending count reaching zero.
Capacity is the ring size and the receive plan, fixed at `start()`; a full ring is a
setup error reported by the plan printer (SDF balance), never a runtime branch. Check: the
audit's conditional enumeration on the hot flows contains only: ring empty, verbs
return, pending == 0.

**H5. Dependent loads are last in line.** Any load whose address depends on a prior
load — reference-count words by row, lifecycle records, status words, observer state,
Swift-side bindings — occurs after the hot stores of its event have been issued, on the
cold pass; the hot store is never conditioned on it. Check: in every hot function the
first store to a ring precedes every dependent load in program order (read the source);
in a trace the doorbell time is independent of cold-pass backlog.

**H6. Contiguity by construction, not by placement.** A transfer's chunks are posted in
chunk order on one queue (D5) into `receive_record` pages that are consecutive
(`first[t] + k·block`); the consumer's operand is therefore contiguous and
`operand_array` holds its address at `start()`. No page-table lookup, permutation,
descriptor, compact list, alias slot or asynchronous placement copy exists on any path.
Interleaving from multiple peers is different queues, not shuffled pages. Check: the
number of runtime page-table reads on every path is zero; `mesh_page[]` and any
successor exist only in `start()`.

**H8. Deletions this group requires** (from the ontology audit; each is a named
structure whose only function was permission or resolution): `mesh_wire_tag` beyond the
row/frame word the WR already carries; `buffer.definition`, `buffer.invocation`,
`buffer.mapping` and `mesh_buffer_pages()`; `receive.definitions[]`, `receive_binding.rows[]`,
`receive.active[]`; the notice levels (`mesh_notice_reader`, summary words) → `fire_ring`;
`mesh_use` + `program[]` → `use_record`; `mesh_join`, `matches[]`, `join_free`,
`join_rows`, `mesh_hash`, `mesh_call_match/join` → `call_record[frame]`; per-function
`available[]`/`values[slot]` → `slot = frame`; `mesh_operand.{row,invocation,pages,…}` +
`mesh_operand_address` → a pointer; `rearm` allocation → pre-built command buffers per
frame; the Swift `copies/chunks/sources/targets` placement copy. Check: none of these
symbols has a runtime reader.

### R — segmentation recovery (cable pulls, replugs, port moves; between NFEs only)

Library backlog; not a prerequisite or gate for Goal A/B.

Facts this group is written against: TB5 RDMA device names follow the interface, so a
replug can present the same peer on a new `rdma_enN`; the neighbour cache is empty after
a replug (`RDMA-RULES.md`); a send completion is not delivery (D12); UC has no repair
(transport-boundary.md); a wedged verbs call is unkillable. The operator's rule: a peer
pair that loses its link is the same two peers with a new indexed connection; they
re-establish who they are and what they will do; nothing inside an NFE is recovered.

**R1. Peer identity is independent of the link.** Signature: the pairing exchange carries `(node, boot_nonce, program_epoch)`; a re-pair with the same `node` on any device is the same peer with a new `connection_index`; `Topology.Link` gains `connection: Int`. Ref: TN3205 out-of-band metadata; `RDMA-RULES.md` versioned exchange. Check: `observe()` after a port move shows the same peer node with a new device and `connection+1`; no code path keys a peer by device name or GID.

**R2. Loss is observed at the completion queue and the control socket, never by a data-path timer.** Signature: a failed work completion or a control-socket error puts the link in `MESH_STOPPED` with the errno in `port.code`; every instance whose plan uses that link concludes `Result.link(peer:code:)` on its next `result()` (one load); its slots are released by the ordinary refcount events (no leak, X5). Ref: fail-stop; end-to-end; I12. Check: the source audit finds no clock read on any data-path function (only the pairing exchange reads time); the killed-peer run (row 16) concludes within one completion.

**R3. Re-pairing is a bounded loop that runs while the bridge is up, regardless of clients** (subsumes 19j). Signature: the link controller repeats `{re-enumerate devices matching the configured peer, warm the neighbour cache (`ping6 ff02::1%iface`), re-create QP/CQ if the device context was lost, exchange (R1), post the planned receives}` with one bounded deadline per attempt and a fixed pause between attempts; the control listener is closed and reopened per attempt (no stale backlog). Ref: `RDMA-RULES.md` (bound every syscall; never SIGKILL); TN3205. Check: `link_run` has no `if(!transfers) return`; a follower started 5 minutes before rank 0 pairs when rank 0 attaches; `bin/mesh-status.sh` shows `attempt` counts.

**R4. A realized program survives a link loss.** Signature: after `MESH_STOPPED → MESH_PAIRED` on the same peer, `Mesh.submit(next)` succeeds with no new `Mesh(…)`, no `start()`, no re-registration by the client: sections, bindings, planned pages and send-edge tables are unchanged; the bridge re-posts the planned receives for the new connection from the same tables. Instances in flight at the loss are the ones that concluded in R2. Ref: MPI-4 persistent collectives (a persistent request outlives a failed start); Pathways. Check: the driver's recovery path performs no re-realization (read it); the demonstration in R7 shows `submit` succeeding after re-pair with the same program object.

**R5. Port moves are configuration-free.** Signature: the bridge config names peers, not devices: `links=('*,1,…')` or a device list per peer; at attempt time the controller selects any `PORT_ACTIVE` device whose neighbour table contains the peer's link-local address. Placement routes (T2) name pairs, never devices. Ref: `RDMA-RULES.md` "warm the neighbour cache before RTR". Check: moving the cable to another port on either node re-pairs with no config edit and `observe()` shows the new device.

**R6. Partition with P > 2 is a topology change, not an error storm.** Signature: each side's `Result.topology(lost:, gained:)` (T5) names exactly the peers it lost; instances needing a lost peer conclude failed; the driver re-realizes on `observe()`'s reachable subgraph between NFEs; ownership never migrates on its own (I9). Check: T5's cable-pull run on a ring names one link on each side and nothing else.

**R7. Demonstration and number.** Signature: on the pair, pull the cable during an NFE: the NFE concludes `Result.link` within one completion on both sides; replug (same or other port): the next `submit` completes with no process restart, no bridge exit, no U-state process; report time-to-repair (cable in → paired) and time-to-first-completed-NFE-after-repair. Check: bridge pids unchanged before/after; both numbers in `output_data/mesh_recovery/`; repeated three times.

**X9. Presence stamps are device-readable; consumers may be resident kernels.** Optional. Signature: `TensorPart.stamp` — the address (in the same shared mapping the GPU sees via `bytesNoCopy`) of the section's per-instance presence word; the bridge/worker writes it with a release store after the data is visible (X1). A supplied `TensorFunction.metal` may encode one command buffer for a whole local step and wait *inside the kernel* on the stamps of the sections it consumes (device-scope atomic load loop), or use a Metal 4 queue-side `waitForEvent` that the RX thread signals — either way the dependent work is resident before the tile lands and the crossing costs the wire only. The in-kernel wait is the X2 firing rule executed on the device: presence only, no data-dependent branch, no retry; the GPU watchdog is the fail-stop (`Result.function` on timeout, I12/I19). Ref: MLX `fence_wait` (`MLX_METAL_FAST_SYNCH`); NCCL LL flags; MSCCL++ device semaphores; Gupta et al. 2012 persistent threads; Anukari (spin to keep the Apple GPU clocked). Check: `TensorPart.stamp` exists and `mesh_publish` stores it release-ordered after the payload; a caller example encodes one command buffer per step with in-kernel waits and completes with command buffers per step = O(segments), not O(tiles) or O(layers); measured µs-class arrival-to-consume on the pair.

### F — the scale-out structure as types and satisfaction constraints

Notation (the three layouts of one value, per the Definition): `R` the total `T(X)`,
`S(slice)` a codomain-projection partial `Q'_j T(X)` (disjoint support), `P(term)` a
domain-projection partial `T(Q_i X)` (full support, pending `+`); `N` nodes with rates `r_i` (leader = 1); `K` row tiles; `c` = wire latency +
serialization of one tile's vector; `T_f` the split sublayer's leader-alone time;
`T_rest` the unsplit remainder per layer.


**F2. Crossing count is a static function of the chain.** `crossings(program) = |{ P -> R transitions on the critical path }|`. FFN-only TP: exactly 1 per layer (the norm after the residual); Megatron: 2.

**F3. Tile pipeline.** `tiles : [Range<Int>]` a ragged partition of the rows; every op instance and every transfer edge is per tile; a dependency edge between tiles `t ≠ t'` exists only where the arithmetic reads across rows (attention over one sequence at prefill; none at decode). Satisfied when: the dependency graph produced at `start()` has no inter-tile edge except those, so layer `L+1` on tile `t` may run while tile `t+1` of layer `L` is in the reduce; exposed crossing per layer is `≤ c/K + one tile's compute` in the trace.

**F4. Hops per layer = 1; bytes are chosen to keep it 1 (I25).** Direct push of each node's partial to every consumer (`(N−1)·|vector|·rows·sizeof(elem)` per node, one hop, summed on arrival) whenever `(N−1)·bytes / 10 GB/s < one hop (~12 µs)`, i.e. below ~120 KiB per push — every decode vector (3 KiB per token on E2B, ×B); reduce-scatter + all-gather (`2·(N−1)/N·|vector|…`, Patarasuk–Yuan; two hops) only above that, pipelined per tile so the exposed cost per layer is still one hop. The choice is a pure function of `(bytes, N)` evaluated at `start()` and printed. Satisfied when: the printed plan shows one serial push per layer on the decode critical path and the byte rule for every edge; the bridge's per-link byte counters agree after a run. Not it: bandwidth-optimal RS+AG on a 3 KiB vector (two hops to save 6 KiB); any runtime choice.

**F5. Command buffers per step per node are constant in N and K.** Optional. `cb(step) ≤ segments_per_layer × layers` today (E1c: 3), and `cb(step) = O(1)` with resident consumers (X9: one command buffer per step with in-kernel stamp waits). Satisfied when: the count is printed at `start()` and does not change with `LM_MESH_SIZE` or the tile count; no completion-handler commit sits between a tile's arrival and its consumption.

**F6. Placement is rate-proportional and bound-checked.** `share_i = r_i / Σ r`; `columns_i = round4(share_i · W)`; heads by the grouping rule; `bound(N) = 1 / ((1−f) + f / Σ r)` is context. Satisfied when: `placement.py` writes the files and prints the bound.

**F7. The exposed crossing has no host term.** Optional. `c_exposed = latency_base + tile_bytes / bandwidth + hops · hop_latency`, all wire; the consumer is resident (X9) or already committed; no `waitUntilCompleted`, completion-handler commit, or shared-event wait between arrival and use. Satisfied when: the trace shows arrival-to-first-consuming-kernel-start ≤ 2× the wire estimate.

**F8. Asymptote.** Backlog; not a gate. `T(N) = T_rest + T_f / Σ r + c_exposed / K` per layer; `T(N+1) ≤ T(N)` always (diminishing, never negative); `T(∞) = T_rest + c_exposed / K`. Satisfied when: measured `T(2)` and `T(4)` on the fleet are within 15% of the formula with the rates from `report.py`, and no measured `T(N+1) > T(N)`.

**F9. Memory per node is constant in N.** `arena_i = instances × Σ_sections bytes(tile)`; weights: the split sublayer's slice (`share_i`) plus the replicated remainder. Satisfied when: `start()` prints the arena and it is independent of `N`; the registered span fits the 4 GiB bank rule.

**F10. Failure is per tile-instance and between NFEs.** Library backlog; not a gate for Goal A/B. Link loss concludes the tile-instances in flight over that link (`Result.link`) and nothing else; the program survives (R4); resumption is the next `submit`. Satisfied when: R7's cable pull shows only the in-flight tiles failing and the bridge pids unchanged.

**X10 (superseded by H2/H3; kept for the record shapes).** Signature: `struct mesh_send_record` (per transfer chunk, indexed by the TX ring slot / `wr_id`): `{page, bytes, queue, tag[4], release_row}`; `struct mesh_receive_record` (per posted RECV, indexed by `wr_id`): `{row, stamp_index, use_first, use_count, publish_last_chunk}`; `struct mesh_use_record` (per consumer edge, contiguous per row): `{call, pending_address}`; `struct mesh_call_record` (per (call, instance)): `{operand_array_address, worker, pending, output_rows[k]}`; `struct mesh_instance_record`: `{status_word, first_row, row_count}`. Each `_Alignas(32)`, `sizeof ≤ 32` (or ≤ 64 documented), filled at `start()`, read once per event, never mutated except the counters they name. The receive completion is: `rec = receive_records[wr_id]; store presence[rec.stamp_index]; for u in uses[rec.use_first ..< +rec.use_count]: if(!--*u.pending_address) enqueue(u.call)` — one metadata line, one contiguous use range, one countdown. The send is: `rec = send_records[slot]; ibv_post_send(&rec.wr)` with the WR and SGE pre-built inside the record. Ref: LMAX Disruptor (preallocated ring of fixed records), Monsoon token store (presence next to the operand), Kalia ATC'16 (cache-line-sized WQEs, doorbell batching), NCCL LL (flag co-located with data). Check: the ontology audit's dependent-load count per event equals 1 metadata record + payload on every runtime path; `sizeof` and alignment of every runtime record asserted at compile time; a trace shows completion→consume within the wire + one record load.

**F11. Sequence-split attention is a partial with an associative combine.** Not a requirement (countermanded 2026-09-22). Signature: for a query and a range of KV positions `[p_i, p_{i+1})`, the supplied attention kernel produces, per head, `(m_i, l_i, O_i)` — running max, running sum, unnormalized output — over its positions only; `reduce(parts, to:, using: lseCombine)` with `m = max_i m_i`, `l = Σ_i l_i·e^{m_i−m}`, `O = Σ_i O_i·e^{m_i−m} / l` (flash-decoding split-K; associative, exact in fp32). KV positions are placed **once, when written** (prefill writes each node's positions to that node), proportional to memory bandwidth; K and V never move at decode. On E2B (7 global layers reading full-length caches; 28 sliding layers stay local): crossings per decoded token = 7, bytes per crossing per sequence = 8 heads × (512 + 2) × 4 B = 16.4 KiB, KV bytes read per node per token = its shard. Ref: flash-decoding (Dao et al. 2023); ring/sequence attention (Liu et al. 2023) for the placement; [crossover note §5](e2b-crossover-2026-09-16.md). Check: the plan printed at `start()` shows per node exactly its positions' KV bytes and 7 `reduce` calls per token with 16.4 KiB parts; no `allGather` of K or V anywhere; `T_mesh(B, L) − T_w` in the trace equals `B·L·14 KiB/Σ BW + 7·c` within 15%. Not it: head split on an MQA model (every node would read the whole KV); replicating KV; gathering K/V to the leader.

### E — engine integration and measurement

**E1. The serving step calls Mesh at the Megatron points.** in `metal-microbench`, one file ≤ 300 lines: a Gemma-4 layer where `o_proj` and `down_proj` partials go `call(dot) → reduceScatter(using: add) → call(norm+residual) → allGather`; every other op is `call` on the rank's head/column range with existing kernels bound as `TensorFunction.metal`. Ref: Megatron f/g; Korthikanti 2022; MLX `shard_linear`; Pallas collective matmul (reuse the local kernel). Check: the realized plan for one layer has exactly two collective points (count them in the printed plan); no kernel arithmetic rewritten (kernel files unchanged); solo and n-node are the same binary.

**E2. Measurement.** Pair and solo rates are reported side by side (Goal A period, Goal B decoded tokens/s). Capability sums, bounds and Karp–Flatt are context, never a verdict.

**E3. Depthwise chain at prefill shapes.** Backlog: an optional prefill measurement through E1's segments. Decode TP on the pair is Goal B.

**E4. Rate-proportional placement is the caller's contract.** Signature: the placement file is generated from measured sustained rates (`report.py --rate` values or `bounds()` capability), giving each rank a share of FFN columns and attention heads ≈ `rate_i / Σ rate` (rounded to the kernel groupings: columns % 4, heads % 8 full / % 2 sliding, whole layers where a head count cannot split — E2B full layers stay on the leader); rows split the same way for the residual sections. Ref: HexGen (asymmetric TP shards), Amdahl doc "capability 27 + capability 23", the 09-08 result (11,904/3,456 neurons M5/M4 = 0.78 share gave 1.28× on the 4096-row FFN). Check: `tools/mesh/placement.py --rate m5=27 --rate m4=23 --ratio 0.13 --model E2B` writes the files and prints the resulting shares.

**E5. Prefill through the same binding.** Signature: the E3 graph is realized a second time with `rows = qLen × B` (prompt rows) and the same `bindMeshLayer`, so a prefill NFE runs the FFN/attention TP at the row count where it pays; the prompt broadcast (E3 part 2) feeds it; decode continues on the decode graph. Check: `LM_MESH_PREFILL=1` routes prefill through Mesh; TTFT on a 2048-token prompt is the measured T(2) vs the solo TTFT; the same binary.

**E7. The decode crossover, measured.** Not a requirement (countermanded 2026-09-22). Signature: `report.py --compare --decode --context L --batch B` over the public endpoint, solo on the M5 Max vs the mesh with F11 placement, at `L ∈ {4k, 16k, 64k, 128k} × B ∈ {1, 8}` (prompts of exactly `L` uncached tokens, ≥ 256 decoded tokens timed after the first, V ≥ 2 streams in flight for the mesh); prints, per `(B, L)`: `T_solo`, `T_mesh`, `c` (per-crossing cost from the trace, wire included), the predicted crossover `B·L×(c) = 7c / (14 KiB·(1/BW_leader − 1/Σ BW))`, and `S = T_solo/T_mesh`. Ref: [crossover note](e2b-crossover-2026-09-16.md) §5 table (`c` = 0.28 ms → B·L× = 272k; 25 µs → 24k; 5 µs → 4.9k). Check (I24): solo at reference class or refused; `S(B, L) ≥ 1` at every `(B, L)` with `B·L ≥ B·L×(c)`, and `S(B, L)` within 15% of the model `T_solo/(T_w + B·L·14 KiB/Σ BW + 7c)` everywhere (a mismatch is a hidden mechanism, not noise); `c ≤ 15 µs` — the wire (8–12 µs measured one-way on this stack for ≤ 32 KiB) plus ≤ 2 µs of ours: ≤ 0.5 µs last-GPU-store→spinner, ≤ 200 ns store→doorbell, ≤ 100 ns CQ→stamp, ≤ 1 µs stamp→resident kernel ([budget](e2b-crossover-2026-09-16.md#4-how-fast-a-transaction-should-be--the-floor-stage-by-stage)); a run that clears the crossover with a larger `c` is an H7/F7 failure. Not it: a run at one `(B, L)` chosen past the crossover; a solo baseline from a slow run; per-layer TP decode at small B on an asymmetric pair (I21).

**E1d. The decode layer has one push.** Signature: `bindMeshDecodeLayer(mesh, w, layer:, placement:)` in `metal-microbench`, replacing E1's two-collective layer on the decode path: every rank runs the full attention (all heads, full KV — duplicated) or its head slice, per the caller's placement, the norms, the residual adds and the PLE path in full; runs gate/up/down on its own FFN column slice (`columns_i ∝ BW_i`, E4); its `down_proj` output is a codomain-shaped partial (Definition) which it `send`s directly to every other rank (`3 KiB × B` per push); each rank sums the arriving partials into its residual with a supplied `add` as they land (any order) and proceeds. The lm_head is split by vocabulary tile the same way, each rank pushing its tile's logits (or its local top-k under the sampler's rule) in the same single hop. No `reduceScatter` or `allGather` on the decode path; a head-split attention placement (Goal B's heads [7,1]) adds its one o_proj collective per attention block. Optionally, a rank's step is one resident command buffer (X9/F5) whose consuming kernels spin on the presence stamps of the `N−1` incoming partials. Ref: I25; Megatron FFN column split; MLX `shard_linear` (the local kernel is reused unchanged); NCCL LL (flag co-located with data). Check: the printed plan shows one push per projection pair per layer plus the vocabulary push, each ≤ 120 KiB; attention and norm kernels are the existing ones, run on every rank. Not it: E1's `reduceScatter → norm → allGather` on the decode path; a hop added to avoid running a norm twice.

**E8. Net improvement on the pair at B = 1 decode = Goal B.** Pair decoded tokens/s ≥ 1.035× solo at heads [7,1] and FFN 7:1, with the solo measured next to the pair. The former `S ≥ 1.10`, 0.44 placement, 0.5 ms overhead, 200 µs idle-gap, reference-class and 256-token clauses are countermanded (2026-09-22).

### P — packaging

**P1. Importable and run.** `import Mesh` from a Swift target outside the repo builds with `make -C rdma all` and no environment variables; `bin/mesh-bridge.sh start` is the only prerequisite; the Core ML chain runs on the pair at ABI 43. Ref: MLX `mx.distributed.init()`. Check: a fresh clone on the Mini builds and runs the example by the README command.

**P3. Stable identity, fixed path, one listener.** `rdma/Makefile`: `codesign --force --sign "$(MESH_CODESIGN_IDENTITY)" --identifier io.mesh.bridge` (default identity `Mesh Bridge`, created once per node by `bin/mesh-codesign-identity.sh` in the System keychain; ad-hoc only as a fallback that the check flags); `make -C rdma install-bridge` installs to `$(MESH_PREFIX)/bin` (`/usr/local/mesh/bin`) in place; `bin/mesh-bridge.sh` launches only from there and refuses worktree/build/scratch paths; `install.sh` adds the installed path to the Application Firewall once. Ref: Apple code signing (designated requirement stays constant across builds for a persistent identity); `socketfilterfw`. Check: I20 on both nodes; a rebuild + `mesh-bridge.sh start` produces no firewall or Local Network dialog and no new `socketfilterfw --listapps` entry.

**P2. Pushed.** `git log @{u}..HEAD` empty on both repositories at the end of every turn.

## 5. Not part of the library


- An expression compiler, kernel zoo, scalar DSL, or any numerics inside mesh (I7).
- Migrating a caller's operators into mesh (I8), or a model/layer API in mesh.
- A cost model, auto-placement, or a verb inferred from world size (I9).
- A reuse guard, claim check, ready byte, occupancy check, or handshake (I3, I4).
- A default wait: any `waitUntilCompleted`, semaphore, or poll on the publish path (I2).
- A "gold harness", test suite, or benchmark that imports engine internals (I13, I15).
