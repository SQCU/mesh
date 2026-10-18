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

No row, invariant or check here is an assignment or a gate.

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

The operator's words above govern. Rows are requirements, not assignments, and gate nothing. Where a row's subject
has a document of its own, that document holds its end state: the bridge, [bridge-and-ipc.md](bridge-and-ipc.md);
recovery, [recovery.md](recovery.md); the collectives' API, [rdma/NCCL.md](../rdma/NCCL.md).

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

Restored 2026-10-17 against the restored interfaces (mesh 604f210e deleted it with the Swift module's record; the
module came back in 71bced9e, the Python surface in python/mesh). The importable library is the Swift module `Mesh`
(`swift/Mesh.swift`, `Bounds.swift`, `Topology.swift`) over the C runtime (`rdma/mesh-call.c`, `mesh-dataflow.c`,
`mesh-collective.c`, `mesh.h`, the node's resident bridge `mesh-flow.c`), and its Python surface `mesh` (`python/mesh`)
over libnccl-mesh (`torch.distributed`'s "mesh" backend, rdma/torch-mesh) with torch operands; the kernels and layers
its examples compose (`examples/kernels.py`, `examples/nn.py`) are program type C, not the library (section 5):

```
Mesh(region:, rank:, size:, workers:, inFlight:, placement:, nodes:) // the execution's name; rank r's bridge node
Placement(owners:, routes:, work:, bytes:, cuts:, path:)  // explicit ownership, directed routes, optional bound inputs
Placement.Edge(source, destination)                     // ordered endpoints; route excludes source and includes destination
Mesh.tensor(on:sections:) -> [TensorPart]                 // storage of one term T_i or block Q'_j T(X) (Definition): contiguous sections, each with a rank
Mesh.constant(on:bytes:initialize:) -> TensorPart
TensorFunction.cpu / .metal(device) / .metal(device, operands:) / .resident(device) / .prediction(model, inputs:, outputs:)
mesh_call_operand.load(at:as:)                           // logical element indexing through canonical backing
Mesh.call(function, inputs:, outputs:, on:, worker:)
Mesh.map(function, inputs: [[TensorPart]], outputs:, workers:)
Mesh.send(part, to:, queue:) -> TensorPart
Mesh.broadcast / scatter / allScatter / gather / allGather / allToAll // broadcast, allGather: mesh-collective.c's all-gather
Mesh.reduce(parts, to:, using:, worker:) / reduceScatter(contributions, to:, using:, workers:) / allReduce(…)
                                                          // mesh_compile's reduce, reduce-scatter, all-reduce: each step a send, a REDUCE the combine's call
Mesh.start()                                              // realize storage, bindings, receives, links
Mesh.invocation(slot).submit() -> generation              // admission of one instance
MeshInvocation.result(generation) -> Result<Void, MeshError> // N2: one status load; busy until conclusion
MeshError.busy / .link(peer:code:) / .function(call:code:)
Mesh.syncOnRemoteFill(parts, index:)                      // explicit, diegetic, never default
Mesh.observe() -> Topology                                // the bridges' map
bounds(program, placement, topology, capability) -> Bounds // pure lower bounds; never picks a placement

mesh.Program(backend, *, functions, group, trace)            // SPMD; realize() == start()
Program.tensor(shape, block_shape, dtype, transferable, *, contiguous) -> Tensor  // blocks of Refs (views of backings)
Program.kernel_call(kernel, *, out_shape, grid, in_specs, out_specs, peer)(*operands)  // kernel(program, inputs, outputs, coordinate)
BlockSpec(block_shape, index_map, region_map) / ShapeDtypeStruct(shape, dtype) / Partial(required, terms)
Program.copy(src.on(a), dst.on(b), *, queue) == send == recv / Program.replicate(tensor.on(a), b)
Program.constant(ref, value) / Program.is_constant(ref) / Program.write(ref) / Ref.writable / Program.export(ref) -> Result
mesh.collective: SUM, MAX, MIN, PARTIALS; send, recv, recv_like, broadcast, scatter, gather, all_gather, all_to_all,
                 reduce, reduce_scatter, sum_scatter, all_reduce, all_sum, all_max, all_min, sync_on_remote_fill
                 // a block row one call: AllGatherV, all-to-all, ReduceScatterV, all-reduce, the partials merge; op= a supplied combine
examples/kernels.py (type C): arguments, indices, program_id, select, expression, dot; add (a combine), maximum, minimum,
                 swish, identity, matmul, gather, on_engines(engines)
examples/nn.py (type C): linear, ffn, rmsnorm, embedding, summed_embedding    // K-split partials published tile by tile
Program.report / MeshError(kind busy|link|function, code)
```

Frontier, stated as measured: the module restored in 71bced9e has not been measured. Its commit credits it with the
12B cooperative FFN's 1.145-1.31x over the M5 alone (metal-microbench 43724cb, b6efa23, 11a4a7a); those runs
(2026-09-11/12) went through metal-microbench `tools/mesh/nfe.sh` on the mesh transport before `swift/Mesh.swift`
existed. The partial-tensor library's own frontier is the deleted native executor's (mesh 57628982): the literature
chain paired CPU 675.0 invocations/s against 558.3 at world size 1 (+21%), Metal 172.6 against 155.5; streamed
1,429.2 results/s against 962.9 whole (+48.6%). The Python surface's harnesses reproduce both comparisons
([algorithm-sources.md#streaming-overlap-measurement](algorithm-sources.md#streaming-overlap-measurement)).

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

**S1. The bridge finds its ports and serves them all.** Nothing is configured: the bridge gives each RDMA device it finds an index, and its hellos find the node each one reaches ([bridge-and-ipc.md#ports](bridge-and-ipc.md#ports)). Ref: JACCL hostfile (the full-mesh case of what the map holds). Check: a node with three cabled ports pairs three links (the port table shows each one's phase); the same program runs at world size 2 and 3 with nothing changed. Not it: a hostfile, or a configured peer list.

**S2. TX/RX per link on dedicated threads; drain and refill immediately.** `rdma/mesh-flow.c` `tx_thread(link)`, `rx_thread(link)`; RX reposts before delivering. Ref: NCCL proxy; JACCL per-wire polling. Check: I6. Not it: a progress loop that also runs numerical callbacks.

**S3. Chunked transport, invisible.** `mesh_section_create(ctx, bytes, count, receive, &section)`; internal `C`, `K = ceil(N/C)`. Ref: TN3205 frame matching; llama.cpp PR #26421 (128 KiB stride cost). Check: I10.

**S4. Out-of-band pairing on the control socket; every syscall bounded.** The link hello (node, boot, LID, GID, block size), then each program's READY, PAIRS and POSTED ([bridge-and-ipc.md#protocol](bridge-and-ipc.md#protocol)); `accept/connect/read` each with a deadline ([recovery.md](recovery.md) Q14). Ref: TN3205; `RDMA-RULES.md`. Check: each of those calls in `rdma/mesh-flow.c` has a deadline.

### L — library core

Restored 2026-10-17 (deleted in 604f210e; L5 in 974b5046) against the restored surfaces; each row's text is the
record's, its last sentence the restored surfaces' names.

**L1. Partial tensor storage = sections with ranks.** A `TensorPart(rank, bytes)` is the storage of one term `T_i = T(Q_i X)` (layout `P`) or one block `Q'_j T(X)` (layout `S`) of the Definition; `Mesh.tensor(on:sections:)`. The partial itself is the mathematical term; the storage never carries more than its coordinates. Ref: Pallas BlockSpec. Check: a section is contiguous locally, addressable globally by `(rank, section)`; no hidden dense copy (I4); I23. Restored surfaces: Swift `TensorPart`; Python `Tensor` blocks and `Ref` views (`Program.tensor`, `region`).

**L2. Supplied function over sections.** `TensorFunction.cpu/.metal/.prediction`; `Mesh.call/map`. Ref: Pallas `pallas_call`; StarPU codelets. Check: I7; a CPU, a Metal and a Core ML function each bound through the same `call`. Not it: an expression compiler; a kernel zoo. Restored surfaces: Swift `TensorFunction`; Python `Program.kernel_call` with any supplied callable `kernel(program, inputs, outputs, coordinate)` (the library calls it and nothing else); the expressions evaluated by torch (nothing compiled) are `examples/kernels.py`'s, program type C (section 5).

**L3. Publish on visibility, fire on presence.** completion → `mesh_publish(row)` → for each consumer in the row's contiguous use range: decrement its pending count; zero enqueues it on its worker. No consumer evaluates "are my inputs present". Ref: Monsoon; Realm; TensorFlow Send/Recv; Naiad occurrence counts. Check: I2, I3, I17; `mesh_publish` walks one contiguous range and performs one decrement per use. Restored surfaces: Swift `rdma/mesh-call.c` (a call's pending count); Python `Program` (a token per (writer, reader) edge, a call firing at its last token, design/algorithm-sources.md#program).

**L4. Ten collective verbs as movement + supplied combine.** §2 list; `reduce*` take `using: TensorFunction`. Ref: MPI-4.1 ch. 5; Rabenseifner/Patarasuk–Yuan; Gloo. Check: every verb is `send` + `call(combine)`; movement does no arithmetic; explicit destinations at every world size. Not it: gather-everything-then-sum; any verb inferred from `size`. Restored surfaces: Swift's verbs, bound to mesh-collective.c's compiled programs (`mesh_compile` along the members' trees, `mesh_program_steps_cut`): each step a `send`, a REDUCE step the supplied combine's `call`; gather, scatter and the all-to-all point to point; Python `mesh.collective` (a block row one libnccl-mesh call: movement `ncclMeshAllGatherV` or the all-to-all; a reduction by `SUM`, `MAX` or `MIN` `ncclMeshReduceScatterV` or the all-reduce, by `PARTIALS` the log-sum-exp merge; any other combine movement plus the supplied combine's kernel).

**L5. Partial is a type.** A `P`-layout term of the Definition (full support, pending `+`) is typed: `TensorPart.partial: Bool` set on `reduceScatter/reduce` contributions; `Mesh.call` throws `MeshError.partialOperand(part)` at setup when a non-combine function binds a partial. Ref: DTensor `Partial`; Legion `reduce`; Korthikanti 2022. Check: one `throw` in `call`; no runtime state. Restored surfaces: Python `Partial` (`kernel_call` refuses a partial operand of any kernel that does not declare `combine`, TypeError at recording; `Partial.merge` refuses a contribution counted twice); Swift has no partial type yet.

**L6. Combine is any associative supplied function.** `reduce(…, using:)`; explicit binary tree. A *partial tensor* (Definition) has combine `+` and the exactness of I23; another associative combine is movement plus a supplied function and carries no claim beyond associativity. Ref: `MPI_Op`; Blelloch. Check: no special case for `+` in the movement. Restored surfaces: Swift `reduce(using:)` (the combine's calls placed by the compiled reduce's REDUCE steps); Python `op=` (`SUM`, `MAX`, `MIN` and `PARTIALS` go to libnccl-mesh; any supplied kernel declaring `combine` otherwise).

**L7. Lifetime without explicit free.** internal `mesh_buffer_retain/release`; uses count down on numerical completion, TX completion, destruction; the free pool returns pages unzeroed. Ref: RCU; Disruptor. Check: I5. Restored surfaces: Swift's sections and leases; Python storage freed with its last view (a window lease returned with it, `mesh.lease`).

**L8. Explicit sync as a library call with its counterexample.** `Mesh.syncOnRemoteFill`; `examples/sync-on-remote-fill.swift` `parallel|serial|deadlock`. Check: I11. Restored surfaces: Swift `Mesh.syncOnRemoteFill`, `examples/sync-on-remote-fill.swift`; Python `collective.sync_on_remote_fill` (no bound: a result never filled waits forever), `examples/sync-on-remote-fill.py --mode stream|sync|deadlock|independent`.

### G — genericity

**G1. Contraction is a two-operand `call`; K-partials are `reduceScatter` inputs.** caller: `call(dot, inputs:[x_k, w_k], outputs:[c_k])` per K-panel; `reduceScatter([[c_k]…], to:, using: add)`. Ref: SUMMA. Check: Gram `R(RᵀZ)` and FFN down-projection are the same calls with different operands; the Gram output feeds the next `call` with no copy. Not it: `nn.linear` inside mesh. Restored surfaces: Swift `call` + `reduceScatter(using:)` (examples/gram-chain.swift); Python `examples/nn.py` `linear` (K panels of `kernels.matmul`, `Partial`s summed tile by tile; program type C, outside the library) and `collective.reduce_scatter`.

**G2. Indices as data.** `Mesh.map(function, inputs: [[x], [idx]], …)`; `idx` is a `TensorPart` whose presence gates the map; the supplied function does the indexed read. Ref: Pallas scalar prefetch; `jax.lax.gather`; Megablox. Check: routed-expert and neighbourhood callers exist outside mesh using only `map`; no routing code in mesh. Restored surfaces: Swift `Mesh.map` (examples/indexed-gather.swift); Python `BlockSpec(index_map, region_map)`, and in examples/kernels.py `gather` and an expression's `.at(row, column, mask, other)`.

**G3. Two callers, zero specialization.** `metal-microbench` and `xonotic/solver` both `import Mesh`; mesh imports neither. Check: I8; both build against the same module. Restored surfaces: the Swift module and the Python surface (the xonotic solver was deleted in 57628982; its restoration is open).

### N — instances and lifecycle

**N1. Realize once, submit per instance, unbounded.** `Mesh(…, inFlight:)`; `submit(index) -> Result<Void, MeshError>`; the arena is sized at `start()` as inFlight × bytes-per-instance (SDF balance); an instance's pages return to a per-worker free list when its reference count reaches zero (an event, X5), and `submit` when no instance slot is free returns `MeshError.busy` immediately — never waits, never checks readers. Ref: MPI-4 persistent collectives; Pathways; CUDA graphs. Check: `submit(inFlight + k)` is legal and returns a value; `start()` prints bytes per in-flight instance; no allocation, no reader query, no wait inside `submit`. Restored surfaces: Swift `Mesh(…, inFlight:)`, `invocation(slot).submit()`; Python instances are recorded graph copies, each run again by a host write once its readers finished (capacity one; `Ref.writable`).

**N2. Early conclusion as a value.** `Mesh.result(index) -> Result<Void, MeshError>` is one load of the instance's status word (no scan, no poll loop); `MeshError.link(peer, code)`, `.function(call, code)`. Ref: fail-stop; end-to-end argument. Check: I12; the Core ML chain concludes early on a killed peer and the driver re-realizes. Restored surfaces: Swift `MeshInvocation.result(generation)`; Python `Result.ready`, `Program.report`, `MeshError`.

### T — topology and fleet

**T1. The map is a graph value, observed.** `mesh-plan.h` `struct mesh_link_map` from `mesh_map`: the bridges' records of every node's ports, flooded to every bridge, two nodes linked where each holds a port paired with the other ([bridge-and-ipc.md#ports](bridge-and-ipc.md#ports)). Ref: JACCL hostfile as the full-mesh case; OSPF flooding (RFC 2328 §13). Check: a tree, a ring with spurs and a full mesh run the same program with nothing configured; no code path assumes a link exists.

**T2. Routes are the map's; the bridges forward.** `mesh_map` joins members the links leave apart along the whole mesh's path on fewest links; on each node of the route the program is a relay's two flows, a landed record releasing the next hop's SEND cell ([bridge-and-ipc.md#flows](bridge-and-ipc.md#flows)). Ref: Patarasuk–Yuan (spanning tree suffices); Monsoon. Check: leaf→leaf on a star is two SENDs with no process of the hub taking part; per-cut bytes equal declared traffic.

**T3. Multi-link striping per pair.** `send(part, to:, queue:)` where `queue` selects among a pair's links; chunks round-robin when `queue == .all`. Ref: JACCL `ring_impl` wires; TN3205 10 QPs/device. Check: equal bytes per link of a pair in `report`. Restored surfaces: Swift `send(part, to:, queue:)` (a queue of the bridge's flows); Python `queue=` (a libnccl-mesh communicator of its own, its crossings progressing apart from other queues').

**T4. Bounds before runs.** `bounds(program, placement, topology, capability) -> {compute, memory, cut: {cut: s}, path: s, max}`, pure; `topology` is the T1 value (links, observed), `capability: [node: Capability(rate, bandwidth)]` the caller's sustained rates. Ref: `docs/amdahl_superiority.md` four lower bounds; Hockney; LogGP; COSMA. Check: reproduces the ten-minute table (1+3, 2+17, 1+1). Restored surfaces: `swift/Bounds.swift` `bounds(_:_:_:_:)`, `examples/bounds-table.swift`.

**T5. A change of the map fails the executions it cuts; the next reads the map.** A lost link or relay fails each program across it, its status naming the node; nothing is resumed; the next execution's start reads the map as it is then ([recovery.md](recovery.md), mandates 11 and 13). Ref: fail-stop; Dean–Ghemawat; `RDMA-RULES.md`. Check: pull one cable mid-run on a ring: the failure names the node, the next execution completes on the remaining links, and the replug is in a later map.

**T6. Replicas are placement, not protocol.** `Placement(…, replicas: [section: [rank, rank]])`: a replicated section is produced on every listed rank as an ordinary extra output edge; each consumer is bound at `start()` to ONE replica (the placement's primary for that consumer); loss of the primary's link is a T5 retopology event whose re-realization binds the next replica. No read-time choice, no "first present" branch. Ref: Legion physical instances; end-to-end. Check: no runtime conditional selects among replicas; losing a link changes `bounds()` and triggers T5, not a branch. Restored surfaces: none yet (`Placement` has no replicas).

**T7. Many clients, by lease.** The node's bridge serves every client at once, none aware of another ([bridge-and-ipc.md#clients](bridge-and-ipc.md#clients)): rows and arena pages claimed by compare-exchange, an attach refused (EBUSY) where no slot is vacant, a lease refused as a value (ENOMEM, ENOSPC) where the node cannot give it. Ref: NCCL communicator per job; PMIx. Check: two users' programs run on disjoint rows and queue pairs; one client's exit leaves the other untouched; no `SIGKILL` path.

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

Its definition is [recovery.md](recovery.md): its Q5-Q11 are this group's R1-R7 at their end state, R4 (a realized
program surviving a link loss) dropped by mandate 13.

**X9. Presence stamps are device-readable; consumers may be resident kernels.** Required (2026-10-17: the in-stream crossings are restored and measured; metal-microbench docs/principles.md section 6, the operator 2026-10-16, and docs/measurement.md, mb ea5954e). Signature: `TensorPart.stamp` — the address (in the same shared mapping the GPU sees via `bytesNoCopy`) of the section's per-instance presence word; the bridge/worker writes it with a release store after the data is visible (X1). A supplied `TensorFunction.metal` may encode one command buffer for a whole local step and wait *inside the kernel* on the stamps of the sections it consumes (device-scope atomic load loop), or use a Metal 4 queue-side `waitForEvent` that the RX thread signals — either way the dependent work is resident before the tile lands and the crossing costs the wire only. The in-kernel wait is the X2 firing rule executed on the device: presence only, no data-dependent branch, no retry; the GPU watchdog is the fail-stop (`Result.function` on timeout, I12/I19). Ref: MLX `fence_wait` (`MLX_METAL_FAST_SYNCH`); NCCL LL flags; MSCCL++ device semaphores; Gupta et al. 2012 persistent threads; Anukari (spin to keep the Apple GPU clocked). Check: `TensorPart.stamp` exists and `mesh_publish` stores it release-ordered after the payload; a caller example encodes one command buffer per step with in-kernel waits and completes with command buffers per step = O(segments), not O(tiles) or O(layers); measured µs-class arrival-to-consume on the pair.

### F — the scale-out structure as types and satisfaction constraints

Notation (the three layouts of one value, per the Definition): `R` the total `T(X)`,
`S(slice)` a codomain-projection partial `Q'_j T(X)` (disjoint support), `P(term)` a
domain-projection partial `T(Q_i X)` (full support, pending `+`); `N` nodes with rates `r_i` (leader = 1); `K` row tiles; `c` = wire latency +
serialization of one tile's vector; `T_f` the split sublayer's leader-alone time;
`T_rest` the unsplit remainder per layer.


**F2. Crossing count is a static function of the chain.** `crossings(program) = |{ P -> R transitions on the critical path }|`. FFN-only TP: exactly 1 per layer (the norm after the residual); Megatron: 2.

**F3. Tile pipeline.** `tiles : [Range<Int>]` a ragged partition of the rows; every op instance and every transfer edge is per tile; a dependency edge between tiles `t ≠ t'` exists only where the arithmetic reads across rows (attention over one sequence at prefill; none at decode). Satisfied when: the dependency graph produced at `start()` has no inter-tile edge except those, so layer `L+1` on tile `t` may run while tile `t+1` of layer `L` is in the reduce; exposed crossing per layer is `≤ c/K + one tile's compute` in the trace.

**F4. Hops per layer = 1; bytes are chosen to keep it 1 (I25).** Direct push of each node's partial to every consumer (`(N−1)·|vector|·rows·sizeof(elem)` per node, one hop, summed on arrival) whenever `(N−1)·bytes / 10 GB/s < one hop (~12 µs)`, i.e. below ~120 KiB per push — every decode vector (3 KiB per token on E2B, ×B); reduce-scatter + all-gather (`2·(N−1)/N·|vector|…`, Patarasuk–Yuan; two hops) only above that, pipelined per tile so the exposed cost per layer is still one hop. The choice is a pure function of `(bytes, N)` evaluated at `start()` and printed. Satisfied when: the printed plan shows one serial push per layer on the decode critical path and the byte rule for every edge; the bridge's per-link byte counters agree after a run. Not it: bandwidth-optimal RS+AG on a 3 KiB vector (two hops to save 6 KiB); any runtime choice.

**F5. Command buffers per step per node are constant in N and K.** Required (2026-10-17, as X9). `cb(step) ≤ segments_per_layer × layers` today (E1c: 3), and `cb(step) = O(1)` with resident consumers (X9: one command buffer per step with in-kernel stamp waits). Satisfied when: the count is printed at `start()` and does not change with `LM_MESH_SIZE` or the tile count; no completion-handler commit sits between a tile's arrival and its consumption.

**F6. Placement is rate-proportional and bound-checked.** `share_i = r_i / Σ r`; `columns_i = round4(share_i · W)`; heads by the grouping rule; `bound(N) = 1 / ((1−f) + f / Σ r)` is context. Satisfied when: `placement.py` writes the files and prints the bound.

**F7. The exposed crossing has no host term.** Required (2026-10-17, as X9). `c_exposed = latency_base + tile_bytes / bandwidth + hops · hop_latency`, all wire; the consumer is resident (X9) or already committed; no `waitUntilCompleted`, completion-handler commit, or shared-event wait between arrival and use. Satisfied when: the trace shows arrival-to-first-consuming-kernel-start ≤ 2× the wire estimate.

**F8. Asymptote.** Backlog; not a gate. `T(N) = T_rest + T_f / Σ r + c_exposed / K` per layer; `T(N+1) ≤ T(N)` always (diminishing, never negative); `T(∞) = T_rest + c_exposed / K`. Satisfied when: measured `T(2)` and `T(4)` on the fleet are within 15% of the formula with the rates from `report.py`, and no measured `T(N+1) > T(N)`.

**F9. Memory per node is constant in N.** `arena_i = instances × Σ_sections bytes(tile)`; weights: the split sublayer's slice (`share_i`) plus the replicated remainder. Satisfied when: `start()` prints the arena and it is independent of `N`; the registered span fits the 4 GiB bank rule.

**F10. Failure is per execution and between NFEs.** A link's loss fails the executions in flight over it, naming the node, and nothing else; nothing is resumed, and the next execution runs on the map as it is then ([recovery.md](recovery.md) Q1, Q3). Satisfied when: Q11's cable pull shows only those executions failing and the bridge pids unchanged.

**X10 (superseded by H2/H3; kept for the record shapes).** Signature: `struct mesh_send_record` (per transfer chunk, indexed by the TX ring slot / `wr_id`): `{page, bytes, queue, tag[4], release_row}`; `struct mesh_receive_record` (per posted RECV, indexed by `wr_id`): `{row, stamp_index, use_first, use_count, publish_last_chunk}`; `struct mesh_use_record` (per consumer edge, contiguous per row): `{call, pending_address}`; `struct mesh_call_record` (per (call, instance)): `{operand_array_address, worker, pending, output_rows[k]}`; `struct mesh_instance_record`: `{status_word, first_row, row_count}`. Each `_Alignas(32)`, `sizeof ≤ 32` (or ≤ 64 documented), filled at `start()`, read once per event, never mutated except the counters they name. The receive completion is: `rec = receive_records[wr_id]; store presence[rec.stamp_index]; for u in uses[rec.use_first ..< +rec.use_count]: if(!--*u.pending_address) enqueue(u.call)` — one metadata line, one contiguous use range, one countdown. The send is: `rec = send_records[slot]; ibv_post_send(&rec.wr)` with the WR and SGE pre-built inside the record. Ref: LMAX Disruptor (preallocated ring of fixed records), Monsoon token store (presence next to the operand), Kalia ATC'16 (cache-line-sized WQEs, doorbell batching), NCCL LL (flag co-located with data). Check: the ontology audit's dependent-load count per event equals 1 metadata record + payload on every runtime path; `sizeof` and alignment of every runtime record asserted at compile time; a trace shows completion→consume within the wire + one record load.

**F11. Sequence-split attention is a partial with an associative combine.** A requirement (2026-10-17: metal-microbench docs/principles.md section 4, "split-KV attention for long context (Flash-Decoding [Dao et al. 2023])"; the log-sum-exp combine is a collective of the library, libnccl-mesh, rdma/NCCL.md, and in the Python surface `collective.reduce(..., op=PARTIALS)` and `all_reduce(..., op=PARTIALS)`, ncclMeshReducePartials and ncclMeshAllReducePartials over a block read as nccl.h's [m][l][O] through torch_mesh's ProcessGroupMesh.merge_partials, on either path). Signature: for a query and a range of KV positions `[p_i, p_{i+1})`, the supplied attention kernel produces, per head, `(m_i, l_i, O_i)` — running max, running sum, unnormalized output — over its positions only; `reduce(parts, to:, using: lseCombine)` with `m = max_i m_i`, `l = Σ_i l_i·e^{m_i−m}`, `O = Σ_i O_i·e^{m_i−m} / l` (flash-decoding split-K; associative, exact in fp32). KV positions are placed **once, when written** (prefill writes each node's positions to that node), proportional to memory bandwidth; K and V never move at decode. On E2B (7 global layers reading full-length caches; 28 sliding layers stay local): crossings per decoded token = 7, bytes per crossing per sequence = 8 heads × (512 + 2) × 4 B = 16.4 KiB, KV bytes read per node per token = its shard. Ref: flash-decoding (Dao et al. 2023); ring/sequence attention (Liu et al. 2023) for the placement; [crossover note §5](e2b-crossover-2026-09-16.md). Check: the plan printed at `start()` shows per node exactly its positions' KV bytes and 7 `reduce` calls per token with 16.4 KiB parts; no `allGather` of K or V anywhere; `T_mesh(B, L) − T_w` in the trace equals `B·L·14 KiB/Σ BW + 7·c` within 15%. Not it: head split on an MQA model (every node would read the whole KV); replicating KV; gathering K/V to the leader.

### P — packaging

**P3. Stable identity, fixed path, one listener.** `rdma/Makefile`: `codesign --force --sign "$(MESH_CODESIGN_IDENTITY)" --identifier io.mesh.bridge` (default identity `Mesh Bridge`, created once per node by `bin/mesh-codesign-identity.sh` in the System keychain; ad-hoc only as a fallback that the check flags); `install.sh` installs `mesh-flow` to `/usr/local/mesh/bin` when it differs, and launchd's `io.mesh.bridge` runs it from there. Ref: Apple code signing (designated requirement stays constant across builds for a persistent identity); `socketfilterfw`. Check: I20 on every node; an update produces no firewall or Local Network dialog and no new `socketfilterfw --listapps` entry.

**P2. Pushed.** `git log @{u}..HEAD` empty on both repositories at the end of every turn.

## 5. Not part of the library


- An expression compiler, kernel zoo, scalar DSL, or any numerics inside mesh (I7).
- Migrating a caller's operators into mesh (I8), or a model/layer API in mesh.
- The layering, the operator's (2026-10-17): "program type A in bridge, program type B in meshlib, program type C in
  meshlib/examples". Type A is the node's resident bridge (rdma/mesh-flow.c), type B the library (the Swift module
  `Mesh`, the C runtime, libnccl-mesh and its Python surface `python/mesh`), type C its examples. The Python
  surface's kernels and layers (`examples/kernels.py`, `examples/nn.py`: compositions over torch, restored 2026-10-17
  from the 09-14 library on the operator's "1: SOME FUCKING HIGHER ORDER FUNCTIONS FOR PARTIAL TENSORS",
  collective-goals-2026-09-14.md) are type C, so I7 and I8 hold of the library: it calls a supplied callable and
  nothing else, its reductions are libnccl-mesh's or a supplied combine, and a kernel takes a `Partial` where it
  declares itself a combine.
- A cost model, auto-placement, or a verb inferred from world size (I9).
- A reuse guard, claim check, ready byte, occupancy check, or handshake (I3, I4).
- A default wait: any `waitUntilCompleted`, semaphore, or poll on the publish path (I2).
- A "gold harness", test suite, or benchmark that imports engine internals (I13, I15).
