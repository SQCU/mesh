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

## The machine

Mesh is a computer: "*what* [we are programming] is a kind of computer, not *how*, or *why*" (the operator,
2026-09-15, above). The machine as this document described it:

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
  wire latency per hop; nothing else in a trace.

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
   that are *added*: `P(term)` in the F notation (`cut(W, rows)`:
   `XW = Σ_i (XQ_i)W`). Codomain projections `Q'_j`, `Σ_j Q'_j = I`, give terms
   `Q'_j T(X)` with *disjoint* support: `S(slice)` (`cut(W, columns)`:
   `XW = Σ_j XWQ'_j`) — the same sum, but each term is zero outside its block, so only
   the block moves and "reduce" is placement into the block. `S` and `P` are the two
   partial layouts of one value and `R` is the total; `reduceScatter : P → S` is the sum
   restricted to a block, `allGather : S → R` is the sum of disjoint-support terms, i.e.
   a copy.

Two corollaries. Given the total `T(X)` alone, the domain-projection split is
`S(t) = (T Q_i T^{-1} t)_i` and exists when `T` is invertible; the program does not execute it — partials originate where the domain shard
is, and the only operation ever applied to a total is a codomain projection (an index
range: free). A nonlinear `f` has `f(Σ_i T_i) ≠ Σ_i f(T_i)`: the sum completes before any
nonlinear function, and that completion is the crossing (F2).

In the machine a partial is three integers fixed at `start()` and carried by the receive
record: which sum it belongs to (the consumer's row), which term `i`, and which codomain
coordinates its bytes are (a chunk range). Its combine is `+`. Ragged sizes are chunk
ranges; nothing else about a partial exists at runtime.

## 0. Status of this document

Where a row's subject has a document of its own, that document describes it: the bridge and recovery,
[transport.md](transport.md); the collectives' API, [rdma/NCCL.md](../rdma/NCCL.md).

## 1. Invariants of the design

| id | The design | The operator's words, or the source |
|---|---|---|
| I1 | Two-sided only: `IBV_WR_SEND` into a posted RECV, on UC queue pairs. | TN3205 offers SEND and RECV, and no WRITE, READ or atomics (hellas `-ENOTSUPP`). |
| I2 | A producer publishes when its writes are visible, and returns. | 2026-09-16 (above): "anything taht stops data from being streamed to targets immediately sounds like it's in error and doesn't comport with the standards and requirements of the transport layer lib." |
| I3 | Presence is the receive completion. The data path carries no acknowledgement, handshake, health check, credit protocol or in-band sentinel. | M2 (design/rubrics.md), the operator's pair: lower, "if you have successfully 'asked permission' to send data"; upper, "you could have sent the data instead" (10-19). |
| I4 | Memory mapping, not control flow: no reuse guard, claim check or occupancy check withholds ready work, and page reuse on a fixed ring adds no wait on the hot path. | 2026-09-15 (above): "dependent loads axiomatically cannot sync or guard or induce waits for transfers". |
| I5 | Lifetime is reference counting owned by mesh, with no free in the caller. Release returns pages to the pool without zeroing them. | collective-goals-2026-09-14.md: 2026-09-15, "have a subagent literally write a refcounter which doesn't rely on users of buffers specifically and literally inlining some kind of 'please say im done using the resource' explicit free like we're in c."; 2026-09-14, "remember that garbage collection in this case simply means 'releasing' pages into the 'pool of pages you can write to if you wanna lol' without even needing to zero them lol" |
| I6 | RDMA TX and RX each own a hardware thread per link and share it with nothing; every queue drains and refills at once. | 2026-09-14 (collective-goals-2026-09-14.md): "there is no reason for the send/receive rdma work to share a thread with any other work under any situation for any reason"; "these hardware constraints simply mean that all queues must be drained IMMEDIATELY WITHOUT EXCEPTION WITH NO WAITS SEPARATING QUEUES FROM BEING FILLED AND EMPTIED." |
| I7 | Mesh contains no numerics. Matmul, activation, norm, sum, max, attention and sampling are supplied functions. | 2026-09-14 (collective-goals-2026-09-14.md): "we DO NOT NEED TO IMPLEMENT EVERY SINGLE ALGORITHM EVER INLINE INSIDE OF OUR MESH CODE. we can in fact SEPARATE THE TRANSPORT AND PARTIAL PUBLICATION AND SO ON from the FUNCTION WHICH IS RUN OVER THE TENSOR PARTIALS" |
| I8 | Mesh contains no application: no model, layer, game, policy or caller name in the library. | 2026-10-17 (section 5): "program type A in bridge, program type B in meshlib, program type C in meshlib/examples". |
| I9 | The caller places. The library has no cost model, automatic sharding or topology inference, and infers no collective from rank count or tensor type. | 2026-09-14 (collective-goals-2026-09-14.md): "why are there cost models lol we're talking about a scituation where we specified asnyc producers, async consumers, and all responsibilty lands on the CALLER to supply a MESH AND TENSOR PLACEMENT CONFIG which is USEFUL instead of USELESS."; 2026-09-15: "note that peer>1 prohibits the use of type inference of broadcasts or alls rather than defined scatters and defined gathers" |
| I10 | Transport fragmentation is invisible: one request does not bound a section's bytes, and chunking changes no shape, partition, function or verb. | 2026-09-15 (collective-goals-2026-09-14.md): "remember that a streaming transportmessagesize transport layer implementation should be totally transparent to all users of a send or receive api call exposed through our mesh api; this is something which is purely internal and have no effect on any caller or user of this api at any point in the future." |
| I11 | Explicit synchronization is a library call a caller makes (python/mesh's `collective.sync_on_remote_fill` was one; none is offered since it was deleted), not a default. | 2026-09-14 (collective-goals-2026-09-14.md): "if we want to supply a 'sync_on_remote_fill' barrier, this is library code that needs to be used EXPLICITLY AND DIEGETICALLY IN CALLING CONTEXTS, it can NEVER be a default behavior." |
| I12 | A failure is a status value out of band: an operation concludes early on a link error, and the driver invokes again at the data layer. | 2026-09-10 (design/pages-and-functions.md): "error status returns are allowed by the specifications we use... but it can never be synchronous or blocking or consumed by a callgraph itself... it can only be passed monadically through a metadata channel..." |
| I13 | Goal A's period and Goal B's decoded tokens/s ratio on the two-node program were September 2026's measures ([collective-goals.md](collective-goals.md), history). | Quantities (design/rubrics.md), 10-20: "meausrements which do not contian those specific values are lesser than measurements which do". |
| I14 | The collective is Megatron-LM TP2 (Shoeybi et al. 2019) with a reduce of row-split partials. | 2026-10-22 (design/extents-and-liveness.md): "connection to the prior literature is important for understanding what are choices and what are strictly dominating superior methods we must use". |
| I15 | — | 2026-09-15 (above): "the 'deliverables list' is actually a regression test suite in a codebase which, dan luu style, *prohibits* unit tests and hand-written tests." |
| I16 | — | 2026-09-15 (above): "bad code is big and does nothing. bad code is bad because revising it is more complicated than deleting it. doing nothing while staying big takes proactive work." |
| I17 | Indexed dataflow on the hot path. Logical rows, consumer ranges, storage capacity and backend bindings are realized before execution. The runtime's page mappings, queues, counts and operands are compact arrays; arrival-derived page indices and asynchronous placement take interleaved arrival. Physical arrival order defines no tensor coordinate and no collective's semantics. | 2026-09-16 (above): "you can precompile any kind of binpacking solver and indexing, reindexing, and virtualization strategy you want but you can't introduce any latency in the receipt or use of data to do it." |
| I18 | Producers publish on visibility and return; dedicated TX/RX threads drain completions and post available work continuously. No barrier, acknowledgement, reuse guard, blocking retry or wait for a particular value holds up independent publication, transfer or consumption. Layout conversion is asynchronous dataflow over its own operands. | 2026-09-15 (above): "i18 was misspecification and described guards or blocks which, ironically, keep showing up in library code as control flow that prevents values from being drained or transacted." |
| I19 | Recovery happens between NFEs: a link failure concludes the in-flight instances as status values (N2), and the driver's next `submit` resumes. The bridge's part is [transport.md](transport.md). | 2026-09-15 (above): "recovery only has to happen between NFEs; there is no error recovery guarantee within a NFE." |
| I20 | `rdma/Makefile` signs the bridge with the identity `Mesh Bridge` and the identifier `io.mesh.bridge`, ad hoc where that identity is absent, so a rebuild keeps one designated requirement (Apple code signing). | — |
| I21 | Superseded 2026-09-22 by Goal B (B = 1 decode TP on the M5 Max + M4 Pro pair, heads [7,1], FFN 7:1). | — |
| I22 | One shm→L1 per event. Each runtime decision (a completion, a publication, a submit, a consume, a result read) reads one contiguous, aligned metadata record, a 32-byte block (or one 64/128-byte cache line where the platform line is wider), precomputed at `start()` and indexed by the event's integer (`wr_id`, row, instance, call index). The record holds everything the event needs (addresses, lengths, queue, tag words, consumer range bounds, pending-count location, stamp address): no second dependent load, table of tables, struct of pointers, page-table lookup, hash probe or string key. Algorithms are written as wide-line loads and stores over these records. | 2026-09-15 (above): "remember that we are allowed to describe widecachelineload algorithms instead of e.g. python algorithms and specifically require that metadata and stupid bullshit be loaded in contiguous 32x8bit blocks or analogous"; "1 cache line read per event maximum" |
| I23 | Partials are exact and schedule-free (Definition). Every `(term i, chunk c)` of every reduction is added once into one destination range; no arrival order, grouping, chunk grain or tile size changes the arithmetic, and no runtime path splits a total into domain partials (`T Q_i T^{-1}`) or reconstructs a partial from anything but its own domain shard. | the Definition |
| I24 | The crossover: [e2b-crossover-2026-09-16.md](e2b-crossover-2026-09-16.md), history. | — |
| I25 | The cost of a layer is the number of serial pushes on its critical path times the floor of one push (~12 µs), plus the split compute. (a) On the decode path, one serial push per layer: every node computes in full what costs less duplicated than a hop (norms, residual adds, embeddings, attention at short context, sampling), computes its slice of what is split (FFN columns; the vocabulary tiles, since the lm_head at 0.8 GB is 1.3 ms, its slice's logits/top-k pushed in the same hop), and pushes its partial directly to every consumer (Definition: a codomain-shaped term, summed locally on arrival, in any order). (b) Reduce-scatter + all-gather (two hops) where `(N−1)·bytes / 10 GB/s` exceeds one hop latency, i.e. prefill-class tiles, pipelined per tile (F3) so the exposed cost is one hop per layer; chosen at `start()` from bytes and `N`. Head-split attention adds one o_proj collective per attention block (standard Megatron); duplicated or head-split attention is the caller's placement. | — |

## 2. Public surface

The importable library is libnccl-mesh (rdma/NCCL.md) through `torch.distributed`'s "mesh" backend (rdma/torch-mesh),
its partial tensors PyTorch's own (DTensor `Partial`, lowered AOT by the FX path: docs/partial-tensors-and-fx.md), with
rdma/mesh.py its binding and mesh_mpi.py its NumPy calls. The Python surface restored on 2026-10-17 was deleted on 2026-10-30. It was python/mesh, a
Pallas-spelled block DSL over torch_mesh; nothing imported it, and `import mesh` never reached it, because mesh.pth
puts rdma/ on the path and rdma/mesh.py answers first. Its Swift twin went in a8952de9, and the kernels and layers its examples composed (examples/kernels.py, examples/nn.py)
on 2026-10-22.

Frontier, stated as measured: the partial-tensor library's own frontier is the deleted native executor's (mesh
57628982): the literature
chain paired CPU 675.0 invocations/s, Metal 172.6; streamed 1,429.2 results/s against 962.9 whole (+48.6%). The
Python surface's harnesses that reproduced both were deleted 2026-10-22; the records stay
([algorithm-sources.md#streaming-overlap-measurement](algorithm-sources.md#streaming-overlap-measurement)).

## 3. References

| Construct | Reference | What it supplies |
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

The transport (the bridge, its pages, links and failures) is design/transport.md; nothing here describes it.

Each gives its signature, its reference and the restored surfaces.

### L — library core

Restored 2026-10-17 (deleted in 604f210e; L5 in 974b5046) against the restored surfaces; each row's text is the
record's, its last sentence the restored surfaces' names. Those surfaces are history now: the Swift module went in
a8952de9, and python/mesh on 2026-10-30 (section 2). On the PyTorch path a partial is DTensor `Partial`.

**L1. Partial tensor storage = sections with ranks.** A `TensorPart(rank, bytes)` is the storage of one term `T_i = T(Q_i X)` (layout `P`) or one block `Q'_j T(X)` (layout `S`) of the Definition; `Mesh.tensor(on:sections:)`. The partial itself is the mathematical term; the storage never carries more than its coordinates. A section is contiguous locally and addressed globally by `(rank, section)`. Ref: Pallas BlockSpec. Restored surfaces: Python `Tensor` blocks and `Ref` views (`Program.tensor`, `region`).

**L2. Supplied function over sections.** `TensorFunction.cpu/.metal/.prediction`; `Mesh.call/map`. Ref: Pallas `pallas_call`; StarPU codelets. Restored surfaces: Python `Program.kernel_call` with any supplied callable `kernel(program, inputs, outputs, coordinate)` (the library calls it and nothing else); the expressions evaluated by torch (nothing compiled) were examples/kernels.py's, program type C (section 5), deleted 2026-10-22.

**L3. Publication on visibility, firing on presence.** completion → `mesh_publish(row)` → for each consumer in the row's contiguous use range: decrement its pending count; zero enqueues it on its worker. No consumer evaluates "are my inputs present". Ref: Monsoon; Realm; TensorFlow Send/Recv; Naiad occurrence counts. Restored surfaces: Swift `rdma/mesh-call.c` (a call's pending count); Python `Program` (a token per (writer, reader) edge, a call firing at its last token, design/algorithm-sources.md#program).

**L4. Ten collective verbs as movement + supplied combine.** §2 list; `reduce*` take `using: TensorFunction`. Every verb is `send` + `call(combine)`; movement does no arithmetic. Ref: MPI-4.1 ch. 5; Rabenseifner/Patarasuk–Yuan; Gloo. Restored surfaces: Swift's verbs, bound to mesh-collective.c's compiled programs (`mesh_compile` along the members' trees, `mesh_program_steps_cut`): each step a `send`, a REDUCE step the supplied combine's `call`; gather, scatter and the all-to-all point to point; Python `mesh.collective` (a block row one libnccl-mesh call: movement `ncclMeshAllGatherV` or the all-to-all; a reduction by `SUM`, `MAX` or `MIN` `ncclMeshReduceScatterV` or the all-reduce, by `PARTIALS` the log-sum-exp merge; any other combine movement plus the supplied combine's kernel).

**L5. Partial is a type.** A `P`-layout term of the Definition (full support, pending `+`) is typed: `TensorPart.partial: Bool` set on `reduceScatter/reduce` contributions; `Mesh.call` throws `MeshError.partialOperand(part)` at setup when a non-combine function binds a partial. Ref: DTensor `Partial`; Legion `reduce`; Korthikanti 2022. Restored surfaces: Python `Partial` (`kernel_call` refuses a partial operand of any kernel that does not declare `combine`, TypeError at recording; `Partial.merge` refuses a contribution counted twice).

**L6. Combine is any associative supplied function.** `reduce(…, using:)`; explicit binary tree. A *partial tensor* (Definition) has combine `+` and the exactness of I23; another associative combine is movement plus a supplied function and carries no claim beyond associativity. Ref: `MPI_Op`; Blelloch. Restored surfaces: Swift `reduce(using:)` (the combine's calls placed by the compiled reduce's REDUCE steps); Python `op=` (`SUM`, `MAX`, `MIN` and `PARTIALS` go to libnccl-mesh; any supplied kernel declaring `combine` otherwise).

**L7. Lifetime without explicit free.** internal `mesh_buffer_retain/release`; uses count down on numerical completion, TX completion, destruction; the free pool returns pages unzeroed. Ref: RCU; Disruptor. Restored surfaces: Swift's sections and leases; Python storage freed with its last view (a region lease returned with it, `mesh.lease`).

**L8. Explicit sync as a library call with its counterexample.** `Mesh.syncOnRemoteFill`; `examples/sync-on-remote-fill.swift` `parallel|serial|deadlock`. Restored surfaces: Python `collective.sync_on_remote_fill` (bounded by the program's timeout); its example, examples/sync-on-remote-fill.py, was deleted 2026-10-22.

### G — genericity

**G1. Contraction is a two-operand `call`; K-partials are `reduceScatter` inputs.** caller: `call(dot, inputs:[x_k, w_k], outputs:[c_k])` per K-panel; `reduceScatter([[c_k]…], to:, using: add)`. Gram `R(RᵀZ)` and the FFN down-projection are the same calls with different operands. Ref: SUMMA. Restored surfaces: Swift `call` + `reduceScatter(using:)` (examples/gram-chain.swift); Python `collective.reduce_scatter` (examples/nn.py's `linear`, K panels summed tile by tile, program type C, deleted 2026-10-22).

**G2. Indices as data.** `Mesh.map(function, inputs: [[x], [idx]], …)`; `idx` is a `TensorPart` whose presence gates the map; the supplied function does the indexed read. Ref: Pallas scalar prefetch; `jax.lax.gather`; Megablox. Restored surfaces: Python `BlockSpec(index_map, region_map)` (examples/kernels.py's `gather` and `.at`, deleted 2026-10-22).

**G3. Two callers, zero specialization.** `metal-microbench` and `xonotic/solver` both `import Mesh`; mesh imports neither. Restored surfaces: the Python surface (the xonotic solver was deleted in 57628982; its restoration is open).

### N — instances and lifecycle

**N1. Realized once, submitted per instance, unbounded.** `Mesh(…, inFlight:)`; `submit(index) -> Result<Void, MeshError>`; the arena is sized at `start()` as inFlight × bytes-per-instance (SDF balance); an instance's pages return to a per-worker free list when its reference count reaches zero (an event, X5), and `submit` when no instance slot is free returns `MeshError.busy` at once, with no wait and no reader query. Ref: MPI-4 persistent collectives; Pathways; CUDA graphs. Restored surfaces: Swift `Mesh(…, inFlight:)`, `invocation(slot).submit()`; Python instances are recorded graph copies, each run again by a host write once its readers finished (capacity one; `Ref.writable`).

**N2. Early conclusion as a value.** `Mesh.result(index) -> Result<Void, MeshError>` is one load of the instance's status word (no scan, no poll loop); `MeshError.link(peer, code)`, `.function(call, code)`. Ref: fail-stop; end-to-end argument. Restored surfaces: Swift `MeshInvocation.result(generation)`; Python `Result.ready`, `Program.report`, `MeshError`.

### T — topology and fleet

**T1. The map is a graph value, observed.** `mesh-plan.h` `struct mesh_link_map` from `mesh_map`: the bridges' records of every node's ports, flooded to every bridge, two nodes linked where each holds a port paired with the other ([transport.md](transport.md)). Ref: JACCL hostfile as the full-mesh case; OSPF flooding (RFC 2328 §13).

**T4. Bounds before runs.** `bounds(program, placement, topology, capability) -> {compute, memory, cut: {cut: s}, path: s, max}`, pure; `topology` is the T1 value (links, observed), `capability: [node: Capability(rate, bandwidth)]` the caller's sustained rates. Ref: `docs/amdahl_superiority.md` four lower bounds; Hockney; LogGP; COSMA. Restored surfaces: `swift/Bounds.swift` `bounds(_:_:_:_:)`, `examples/bounds-table.swift`.

**T6. Replicas are placement, not protocol.** `Placement(…, replicas: [section: [rank, rank]])`: a replicated section is produced on every listed rank as an ordinary extra output edge; each consumer is bound at `start()` to ONE replica (the placement's primary for that consumer); loss of the primary's link is a T5 retopology event whose re-realization binds the next replica. No read-time choice, no "first present" branch. Ref: Legion physical instances; end-to-end. Restored surfaces: none yet (`Placement` has no replicas).

### F — the scale-out structure as types

Notation (the three layouts of one value, per the Definition): `R` the total `T(X)`,
`S(slice)` a codomain-projection partial `Q'_j T(X)` (disjoint support), `P(term)` a
domain-projection partial `T(Q_i X)` (full support, pending `+`); `N` nodes with rates `r_i` (leader = 1); `K` row tiles; `c` = wire latency +
serialization of one tile's vector; `T_f` the split sublayer's leader-alone time;
`T_rest` the unsplit remainder per layer.


**F2. Crossing count is a static function of the chain.** `crossings(program) = |{ P -> R transitions on the critical path }|`. FFN-only TP: exactly 1 per layer (the norm after the residual); Megatron: 2.

**F3. Tile pipeline.** `tiles : [Range<Int>]` a ragged partition of the rows; every op instance and every transfer edge is per tile; a dependency edge between tiles `t ≠ t'` exists only where the arithmetic reads across rows (attention over one sequence at prefill; none at decode), so layer `L+1` on tile `t` may run while tile `t+1` of layer `L` is in the reduce, and the exposed crossing per layer is `c/K + one tile's compute`.

**F4. Hops per layer = 1; bytes are chosen to keep it 1 (I25).** Direct push of each node's partial to every consumer (`(N−1)·|vector|·rows·sizeof(elem)` per node, one hop, summed on arrival) whenever `(N−1)·bytes / 10 GB/s < one hop (~12 µs)`, i.e. below ~120 KiB per push — every decode vector (3 KiB per token on E2B, ×B); reduce-scatter + all-gather (`2·(N−1)/N·|vector|…`, Patarasuk–Yuan; two hops) only above that, pipelined per tile so the exposed cost per layer is still one hop. The choice is a pure function of `(bytes, N)` evaluated at `start()` and printed.

**F5. Command buffers per step per node are constant in N and K.** `cb(step) ≤ segments_per_layer × layers` today (E1c: 3), and `cb(step) = O(1)` with resident consumers (X9: one command buffer per step with in-kernel stamp waits).

**F6. Placement is rate-proportional and bound-checked.** `share_i = r_i / Σ r`; `columns_i = round4(share_i · W)`; heads by their GQA groups; `bound(N) = 1 / ((1−f) + f / Σ r)` is context. `placement.py` writes the files and prints the bound.

**F7. The exposed crossing has no host term.** `c_exposed = latency_base + tile_bytes / bandwidth + hops · hop_latency`, all wire; the consumer is resident (X9) or already committed; no `waitUntilCompleted`, completion-handler commit, or shared-event wait between arrival and use.

**F8. Asymptote.** `T(N) = T_rest + T_f / Σ r + c_exposed / K` per layer; `T(N+1) ≤ T(N)` always (diminishing, never negative); `T(∞) = T_rest + c_exposed / K`.

**F9. Memory per node is constant in N.** `arena_i = instances × Σ_sections bytes(tile)`; weights: the split sublayer's slice (`share_i`) plus the replicated remainder. `start()` prints the arena.

**X10 (superseded by H2/H3; kept for the record shapes).** Signature: `struct mesh_send_record` (per transfer chunk, indexed by the TX ring slot / `wr_id`): `{page, bytes, queue, tag[4], release_row}`; `struct mesh_receive_record` (per posted RECV, indexed by `wr_id`): `{row, stamp_index, use_first, use_count, publish_last_chunk}`; `struct mesh_use_record` (per consumer edge, contiguous per row): `{call, pending_address}`; `struct mesh_call_record` (per (call, instance)): `{operand_array_address, worker, pending, output_rows[k]}`; `struct mesh_instance_record`: `{status_word, first_row, row_count}`. Each `_Alignas(32)`, `sizeof ≤ 32` (or ≤ 64 documented), filled at `start()`, read once per event, never mutated except the counters they name. The receive completion is: `rec = receive_records[wr_id]; store presence[rec.stamp_index]; for u in uses[rec.use_first ..< +rec.use_count]: if(!--*u.pending_address) enqueue(u.call)` — one metadata line, one contiguous use range, one countdown. The send is: `rec = send_records[slot]; ibv_post_send(&rec.wr)` with the WR and SGE pre-built inside the record. Ref: LMAX Disruptor (preallocated ring of fixed records), Monsoon token store (presence next to the operand), Kalia ATC'16 (cache-line-sized WQEs, doorbell batching), NCCL LL (flag co-located with data).

**F11. Sequence-split attention is a partial with an associative combine.** From the literature (metal-microbench docs/principles.md section 4 lists "split-KV attention for long context (Flash-Decoding [Dao et al. 2023])"; the log-sum-exp combine is a collective of the library, libnccl-mesh, rdma/NCCL.md, and in the Python surface `collective.reduce(..., op=PARTIALS)` and `all_reduce(..., op=PARTIALS)`, ncclMeshRedOpCreatePartials's operator over a block read as rows of [m, l, O], through torch_mesh's ProcessGroupMesh.merge_partials, on either path; the engine's split-KV attention calls the same operator on its own rows, metal-microbench mesh_rank.m issue). Signature: for a query and a range of KV positions `[p_i, p_{i+1})`, the supplied attention kernel produces, per head, `(m_i, l_i, O_i)` — running max, running sum, unnormalized output — over its positions only; `reduce(parts, to:, using: lseCombine)` with `m = max_i m_i`, `l = Σ_i l_i·e^{m_i−m}`, `O = Σ_i O_i·e^{m_i−m} / l` (flash-decoding split-K; associative, exact in fp32). KV positions are placed **once, when written** (prefill writes each node's positions to that node), proportional to memory bandwidth; K and V never move at decode. On E2B (7 global layers reading full-length caches; 28 sliding layers stay local): crossings per decoded token = 7, bytes per crossing per sequence = 8 heads × (512 + 2) × 4 B = 16.4 KiB, KV bytes read per node per token = its shard. Ref: flash-decoding (Dao et al. 2023); ring/sequence attention (Liu et al. 2023) for the placement; [crossover note §5](e2b-crossover-2026-09-16.md). Head split on an MQA model would have every node read the whole KV.

### P — packaging

**P3. Stable identity.** `rdma/Makefile` (lines 15 and 20) signs the bridge (`mesh-flow`) and its heartbeat (`mesh-heartbeat`) with `codesign --force --sign "$(MESH_CODESIGN_IDENTITY)" --identifier io.mesh.bridge`: the identity `Mesh Bridge` by default, which `bin/mesh-codesign-identity.sh` makes in the System keychain where it is absent, and ad hoc where the signature with it fails. `make -C rdma install` renames each image that changed into the user's mesh checkout's `rdma/` (`install-images`). The bridge is the admin user's launchd agent `io.mesh.bridge`, which runs `rdma/mesh-heartbeat` from that checkout, and the heartbeat opens `rdma/mesh-flow` with the node's configuration (bin/mesh-user.sh; design/transport.md section 2). `install.sh` removes the former system daemon's `/Library/LaunchDaemons/io.mesh.bridge.plist` and `/usr/local/mesh/bin/mesh-flow` where they remain (install.sh:54), and turns the application firewall off (`socketfilterfw --setglobalstate off`, install.sh:84). Ref: Apple code signing (designated requirement stays constant across builds for a persistent identity).

## 5. Not part of the library


- An expression compiler, kernel zoo, scalar DSL, or any numerics inside mesh (I7).
- A caller's operators moved into mesh (I8), or a model/layer API in mesh.
- The layering, the operator's (2026-10-17): "program type A in bridge, program type B in meshlib, program type C in
  meshlib/examples". Type A is the node's resident bridge (rdma/mesh-link.c, built as rdma/mesh-flow), type B the library (the C runtime and
  libnccl-mesh, with its bindings; its Python surface python/mesh deleted 2026-10-30), type C its examples. The Python
  surface's kernels and layers (examples/kernels.py, examples/nn.py: compositions over torch, restored 2026-10-17
  from the 09-14 library on the operator's "1: SOME FUCKING HIGHER ORDER FUNCTIONS FOR PARTIAL TENSORS",
  collective-goals-2026-09-14.md) were type C, deleted 2026-10-22 with the other test code; I7 and I8 hold of the library: it calls a supplied callable and
  nothing else, its reductions are libnccl-mesh's or a supplied combine, and a kernel takes a `Partial` where it
  declares itself a combine.
- A cost model, auto-placement, or a verb inferred from world size (I9).
- A reuse guard, claim check, ready byte, occupancy check, or handshake (I3, I4).
- A default wait: any `waitUntilCompleted`, semaphore, or poll on the publish path (I2).
- A "gold harness", test suite, or benchmark that imports engine internals (I13, I15).
