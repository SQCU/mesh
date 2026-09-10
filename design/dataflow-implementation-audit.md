# Dataflow implementation audit and performance obligations

September 9, 2026. This continues the record introduced by `d8d69cc` and the
caller rule committed in `7e4995e`. Commit identifiers here are historical
citations, never deployment selectors. The binding algorithm remains
[pages and functions](pages-and-functions.md); this audit corrects claims
about its literature and evidence, and records unfinished implementation work.

## What is mature, and what the citations actually establish

Indexed operand matching, publication ordered after writes, bounded storage
reuse, scatter/gather, and pipelined reduction are established engineering.
Implementing them does not require inventing an application protocol that
announces facts already present in the page table. The repository's stronger
choice is to make that table the sole authority for numerical readiness.
A published system sharing a mechanism is not evidence that it implements
this exact combination, nor a proof of its performance on Metal.

| Primary source | Established mechanism | Limit of the citation |
|---|---|---|
| Papadopoulos and Culler, [Monsoon: An Explicit Token-Store Architecture](https://www.cs.cmu.edu/~18742/papers/Papadopoulos1990.pdf), ISCA 1990, §§2–3 | Compiler-addressed operand slots and presence-bit transitions implement tagged operand matching without an associative search. | Monsoon has activation frames, token queues and issue machinery. It does not demonstrate this host page-table scan or Metal command-buffer coalescing. It is a realization of the firing principle. |
| NVIDIA, [NCCL LL source](https://github.com/NVIDIA/nccl/blob/master/src/device/prims_ll.h) and [LL128 source](https://github.com/NVIDIA/nccl/blob/master/src/device/prims_ll128.h) | Receivers inspect flags associated with transferred data. | NCCL also has steps, connection state, flow control and waits. Its GPU polling is not permission to submit a Metal command that waits on a remote node. The old `src/collectives/device/` citation is stale. |
| von Eicken, Culler, Goldstein and Schauser, [Active Messages](https://www.cs.cmu.edu/~seth/papers/isca92.pdf), ISCA 1992, §2 | Short arrival handlers integrate communication with computation and reduce software overhead. | This is a message-handler mechanism, not a literal modern one-sided RDMA payload-stamp protocol. The earlier blanket attribution of RDMA counter completion to this paper was too strong. |
| Bauer, Treichler, Slaughter and Aiken, [Legion](https://theory.stanford.edu/~aiken/publications/papers/sc12.pdf), SC 2012 | Logical regions describe dependencies and locality; mapping and execution are runtime responsibilities. | Legion contains task scheduling, mapping and region metadata. Region dataflow is relevant prior art, not evidence that these systems have only a literal page table. The SC 2012 paper is Legion; it is not a joint Legion/Realm paper. |
| Murray et al., [Naiad: A Timely Dataflow System](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/naiad_sosp2013.pdf), SOSP 2013, and its companion progress proofs | Timestamped dataflow and distributed progress tracking. | Naiad tracks outstanding work and messages. Timestamps do not abolish progress state. It does not prove that a local stamp reveals remote consumption without a compiled return dependency. |
| Patarasuk and Yuan, [Bandwidth optimal all-reduce algorithms for clusters of workstations](https://gwern.net/doc/ai/scaling/hardware/2009-patarasuk.pdf), JPDC 69 (2009), 117–124 | Bandwidth bounds and collective algorithms under stated network assumptions. | A per-participant byte bound does not bound routed physical-link traffic on every topology, nor remove startup, contention or dependency depth. |
| Wang et al., [Introducing Async Tensor Parallelism in PyTorch](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487), September 2024 | Decomposed communication and matrix work can overlap; the authors explicitly discuss resource contention and wave quantization losses from smaller matmuls. | Their implementation uses symmetric memory and CUDA P2P/copy engines to avoid disadvantages of NCCL send/recv. The earlier claim that it does exactly this over NCCL was incorrect. Their reported scope is large matmuls and intra-node NVIDIA hardware. |
| Saltzer, Reed and Clark, [End-to-End Arguments in System Design](https://web.mit.edu/saltzer/www/publications/endtoend/endtoend.pdf), TOCS 1984 | Complete integrity checks require endpoint knowledge; lower-level checks cannot substitute for them. | The paper permits lower-level mechanisms for performance. It does not require checking after speculative numerical use. That ordering and replay policy are this repository's choice. |

Dennis (1974), Arvind and Nikhil (1990), StarPU, PaRSEC, OmpSs, MapReduce and
RDDs remain useful historical leads from the original record. They are not
additional proofs of a caller with no secondary readiness state. Korthikanti
et al., [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
(2022 preprint, MLSys 2023), establishes sequence-parallel activation-memory
work; the original record did not identify a passage establishing this exact
row-streamed implementation. Do not turn a bibliography into an assertion
that every listed system is identical, production deployed, or 30–50 years old.

## Trace the actual values and execution

For each configured function `f`, generation `g`, and numerical output row `r`,
configuration supplies the input index sets `I(f,r)`, output index sets,
backend function and storage. Its candidate mask is

```
ready(f,r,g) = AND[q in I(f,r)](stamp(q) == g AND page(q) != ABSENT)
R(f,g) = compact(r, ready(f,r,g) AND writable(f,r,g) AND unclaimed(f,r,g))
```

`writable` comes from canonical mesh lifetime dependencies. `unclaimed` is a
proof obligation, not authorization to allocate a second caller-owned table.
A readiness predicate alone remains true while asynchronous work is executing.
A correct realization must explain how a row is selected at most once before
completion, using the canonical ownership/publication mechanism. Neither
pretending selection is completion nor allocating a new jobs dictionary solves
that obligation within the specified design.

The configured numerical function gathers through the physical page indices,
computes over the selected values and masks, and scatters to its bound output
pages. One scan issues one command buffer for its currently selected rows.
Completion publishes those output rows. A noncontiguous selection is an index
vector, not a reason to invent a queue of row-group jobs. Selection storage and
numerical accumulators have pre-realized lifetimes; they do not independently
authorize execution. A backend tile is an implementation of numerical work,
not automatically a command-buffer boundary.

The following are minimum implementation obligations, inferred from the
binding algorithm and the actual memory/execution interfaces:

- Publication: payload and physical page entry become visible before the
  generation stamp; consumers acquire the stamp before reading the payload.
  CPU acquire/release alone does not establish device or NIC completion;
  the backend completion path must establish that visibility before publication.
- Identity: address, row and generation are distinct. Families advance by their
  configured stride. A generation tag cannot preserve bytes overwritten in place.
- Lifetime: all readers, including deferred hashing, finish before storage is
  reused or zeroed. A downstream stamp proves only the reads that its compiled
  dependency actually covers. Remote progress must return on an existing
  dependency page if that is the reuse proof; a local table cannot infer it.
- Granularity: a reduction may consume matching partial pages incrementally.
  RMSNorm needs the complete feature reduction for each normalized row.
  A configured whole-input attention function needs all its declared inputs;
  partial projections need explicit numerical outputs and lifetimes if realized.
- Ownership: transport completion, page return and reusable invocation state
  belong to canonical mesh. The caller supplies numerical functions and their
  static rows. Core ML completion must publish the corresponding rows through
  its realized backend binding, rather than introduce a caller prediction task.
- Failure: no successful stamp before successful numerical completion. Numerical
  disagreement is an evaluation value; link failure is link status. Endpoint
  acceptance must resolve the chosen integrity policy before exposing a result
  as accepted. Replay requires retained/reconstructible inputs and lifetime proof.

At inspection, `rdma/mesh-pages.c::arrive` publishes its stamp with release
ordering after the entry. `mesh_pages_select` performs acquire reads, compacts
indices, **and writes a caller-owned `consumed` mask at selection time**.
The reduce implementation also has `generation`, `done`, consumed masks and
per-reduce threads. These are actual implementation facts, not evidence that
the final static caller has already been realized.

The unfinished `metal-microbench/reduce_scatter.swift` working tree contains
`scanFunction` and `scanNorm`, but also `Job`, `jobs`, `producing: [Int: Task]`,
`prenormed` completion words and consumed masks. `scanFunction` rolls a mask
back when output storage is not producible. `scanNorm` issues normalization
by contiguous runs. A method named scan does not remove the surrounding
scheduler. This audit leaves that existing source work untouched.

Calling this divergence “lock-based” is imprecise without evidence of a lock
on the dependency path. The demonstrated problem is a second representation
of readiness, with polling tasks and serialized issue boundaries. Removing
mutexes would not fix it. Literal indexed selection plus gather/compute/scatter
must replace the authority of that state, not merely appear inside its loop.

## Record of divergence and the ordinary engineering it failed

The historical entries in [distributed-reduce](distributed-reduce.md#what-was-done-instead-and-why-it-is-forbidden)
remain the incident record. The following adds evidence and dispositions.
The measurement narrative lives in
[metal-microbench's Amdahl record](../../../metal-microbench/docs/amdahl_superiority.md).
Numbers below are inherited observations, not fresh runs or independently
recovered raw traces.

| Divergence | Evidence available | Ordinary requirement and disposition |
|---|---|---|
| `K_DIGEST` control frame and mismatch poisoning link status; specification edited to bless it | Introduction `f47fc4e`, spec text `2188f44`; replacement `460ede9` | Keep integrity result at its endpoint and represent the digest as a page. Editing a requirement to bless its violation is a specification failure as well as an implementation failure. |
| `K_OPEN`/`K_READY`, backoff, abort propagation | Removal `47643cc`; late-client page retention `39afad9` | Correct transport ownership and bounded buffering must preserve pages. Removing an application handshake without fixing the bridge's drops was incomplete. Transport flow control remains necessary. |
| Generation advanced by one rather than family stride | Repair `3beb9dd`; loopback narrative records families failing to reduce | Tagged storage must match the configured generation algebra. A run that never exercises later families cannot establish this property. |
| Contiguous hidden matrix crossed page headers | Amdahl loopback record: 0.945 relative RMS, then 0.000375 after paged indexing | Gather by payload address and index. Digest agreement between peers does not prove model correctness. Copying an old error figure onto a new execution path is invalid evidence. |
| Address arithmetic narrower than configured pool | Link record: `Not enough bits` with a 2.1 M-page pool; repair uses 64-bit geometry | Prove byte-offset range from realized storage geometry, including headers. This is ordinary address arithmetic. |
| Reduce/hash/zero work delayed page movement | `39afad9`, `80294e2`; FFN period reported 18.61 ms, tail still 25 ms | Account for CPU service and memory traffic on the transport path. Moving work to another thread is not proof that contention disappeared. |
| Digest progress blocked numerical issue/reuse | Amdahl record describes a later hash overwriting an unchecked one and a digest gate holding storage | Numerical dependency and checking policy must not acquire accidental edges. Existing digest/reuse paragraphs disagree; resolve against the compiled lifetime graph before claiming conformance. |
| Fixed groups, split FFN calls, per-job phases and completion words | Reported ~2,000 command buffers and 1,443 ms/NFE; partial rewrite still carries caller state | Preserve efficient local work, coalesce the selected rows, and charge all launch/scan/copy costs. Overlap alone is insufficient evidence of speedup. |

These failures are technically immature in specific, reviewable ways: incorrect
indexing, incorrect generation arithmetic, duplicated dependency authority,
and performance claims unsupported by the executed configuration. The brazenness
in the record is changing the specification to bless a prohibited mechanism
and reporting replacement code with evidence from its predecessor. Neither
requires guessing the author's motives. Repeating either practice after this
record is an explicit failure to follow the engineering contract.

## Minimum expectations of performance

A conforming implementation must preserve correctness and demonstrate benefit
on the workload it claims to accelerate. Conformance alone does not imply
strictly lower wall time. The phrase “wrong before measurement” is the
repository's architectural rejection criterion; timing still determines the
size and cause of a performance loss.

For fixed model, inputs, output ownership, precision and concurrency, use
sustained rates of the best validated local functions. Useful lower bounds are

```
T >= max(W / sum_i F_i, max_i(M_i / B_i), max_e(Q_e / B_e), L_dependency)
```

Here `W` is useful work, `F_i` the applicable sustained compute rate, `M_i`
local memory traffic, `B_i` usable memory bandwidth, and `Q_e/B_e` the service
required on each physical edge or topology cut. `L_dependency` includes
unavoidable compute, transmission and startup on the critical path. This is
a lower bound, not a claim that all resource limits can be reached together.
Amdahl and Gustafson do not promise that adding any participant reduces latency.

For balanced reduce-scatter/all-gather, `2(n-1)|S|/n` is **sent payload per
participant**; each also receives that amount. Headers and routing add physical
traffic. A ring has `2(n-1)` startup steps in its usual cost model even as its
sent payload approaches `2|S|`. On a degree-three fabric, forwarding and
combining must be mapped explicitly to prove each edge's load. Bounded node
degree or bounded endpoint payload alone does not establish that proof.

Overlap succeeds only when hidden communication exceeds added exposed cost:

```
saved exposed communication > added launch + scan + gather/copy
                              + kernel slowdown + resource contention
```

These are critical-path costs, not independent timers to add indiscriminately.
If a scan repeatedly traverses `E` input entries `P` times, its inspection work
is `O(P E)` even if only a few rows arrive. Page-table scanning is not free.
Coalescing reduces submissions but may delay early rows or change matrix
shapes; preserve the winning local backend and measure both effects. The
PyTorch implementation account above documents this same decomposition cost.

The historical 1,443 ms figure is `3.16 × 457 ms` (single-node reference),
but `1.57 × 917 ms` (quoted barrier latency), not three times that barrier.
Moreover, the Amdahl table labels 917 ms as **two in flight**, while the
streaming report says **one in flight**. Its 462 ms is the two-in-flight
completion period, not response latency. These numbers establish an inferior
reported result relative to the local reference; they do not isolate launch
overhead or establish the speedup of an unwritten replacement. “Exact” in the
narrative means agreement with the previous distributed output: the whole-pass
record reports 0.0138 relative RMS against the local reference.

An acceptable operational comparison records:

1. Same configured numerical functions and backend launch paths; local unsplit,
   local split, distributed barrier, and distributed scan configurations.
   Vary `callRows` and `streamRows` independently where supported. Keep rows,
   shard ownership, weights, precision and in-flight count matched.
2. Warmup and repeated measurements, response latency and completion period
   separately, distribution/spread, device execution state, and numerical error
   against the same local reference. No copied error figures.
3. Command buffers issued, rows per issue, kernel time by shape, scan CPU time,
   actual payload/wire bytes, gather/copy traffic, and exposed exchange tail.
   Aggregate elapsed time alone cannot attribute the regression to submission.
4. Local gain `T_old_local/T_best_local`, additional TP gain
   `T_best_local/T_TP`, and their product. Use the same latency or throughput
   metric for every factor. Rebase whenever a local numerical function improves.

Use the existing client launch/measurement path. No second scheduler, stdin
rendezvous or new repository test harness is needed. The existing record's
promised isolation runs have not been recovered by this audit; their results
remain pending, not successful by implication.

## Handoff

Read this audit before continuing the caller rewrite. First reconcile canonical
at-most-once selection with output lifetime and asynchronous completion; then
express every intermediate as configured rows and bind the existing winning
numerical functions. A second map of jobs, pending predictions or completion
words is not a substitute. Keep all required indexing and visibility machinery
in its canonical owner and make its cost measurable.

The older specification has unresolved textual tensions: digest-as-reuse-proof
versus checking outside numerical dependencies; task/channel wording versus
the static caller; caller-owned selection masks versus table-only authority.
The runtime has not proved these tensions resolved merely by compiling.
This document records them without granting an exception to the operator's
static-scan requirement or changing the specification to fit existing code.

## Refactoring trace: destination ownership (implementation in progress)

The first replacement removes the reduce node's private consumed mask,
generation cursor and completed-group counter. Its static input/output maps
name actual page-table spans. A scan checks input stamps and writable output
storage, claims the destination stamps, and returns compacted numerical indices.
The reduce gathers those operands, adds in FP32 and scatters to the destinations;
publication completes the destination stamps. No reduction cursor advances a
family independently of its input values.

A destination stamp's high bit denotes exclusive write ownership; its remaining
bits identify the generation being written. A completed stamp has no high bit.
Thus a claimed row is neither ready input nor selectable output. Only successful
completion may replace a claimed stamp with its completed generation. This is
canonical page-table ownership, not an independent completion token or array.
Generations are positive and below that reserved bit. Reuse still requires the
compiled downstream lifetime proof; claiming does not shorten an input lifetime.

Static maps and reusable compacted indices are realized before execution. One
scanner owns a configured function, and mapped output spans within it do not
overlap. Multiple asynchronous issues may use different output spans, but may
not reuse the selection buffer until their backend has captured the indices it
needs. The Metal caller's immutable pre-bound row operations can be encoded
immediately; an indexed GPU load needs separately lifetime-bound index storage.
The caller conversion and measurements remain outstanding until recorded here.

### Caller trace and deletion map

The present data path is `embed_paged -> hiddenSlot -> rms_norm_paged ->
normalized[f] -> configured Metal/Core ML function -> partialLocal/partialOut ->
partialIn -> runtime reduce -> reducedOut/reducedIn -> copied tables[f] ->
rms_norm_add_scale_paged -> hiddenSlot`, repeated through the parameter list,
then vocabulary projection. `normalized[f]`, head inputs and logits are real
values whose readiness currently exists outside the page table. The residual
hidden rows also remain live through normalization; recording only the reduced
input does not describe that read.

The present execution path is `queued -> jobs[f] -> Job.stage -> scanFunction /
scanNorm -> embedded/prenormed/headed words or Task completion -> exchanged ->
Job.stage`. That is the path to remove, not an implementation to rename.

| Existing authority/storage | Required replacement |
|---|---|
| `Job.stage`, `previous`, `exchanged`, speculative next-stage issue | Static configured function/input/output maps; generation identity from canonical input stamps. |
| First-free assignment from `queued` to `jobs` | Stable input identity and configured family placement. Both peers must address the same evaluation, independent of completion order. |
| `consumedCalls`, `consumedNorm`, rollback of masks when storage is unavailable | Canonical destination-stamp claims after input readiness and lifetime checks. |
| `embedded`, `prenormed`, `headed`, `producing: [Int: Task]` | Publish the actual embedding, normalized operand and vocabulary result rows; scan their stamps to enable their consumers. |
| Backend branch in `issue` | Bind the backend launch/completion function once during realization. |
| Copied address table and normalization geometry allocated per issue | Direct Metal mapping of mesh's page table; bound geometry and numerical index storage. |
| Contiguous-run normalization command buffers | One command buffer for the configured function's selected indices per scan. |
| Job failure/check counters used to finish or advance numerical stages | Integrity results as endpoint evaluation values; checks cannot authorize intermediate numerical work. |

Core ML has a native completion-handler entry point in the installed SDK:
`MLModel.__prediction(fromFeatures:options:completionHandler:)`. Its spelling
was verified by compiler typechecking. Backend completion can publish actual
result rows without creating a caller task that awaits a prediction. The
backend binding still must retain input/output storage until that completion;
using a callback alone does not establish page-table conformance.

### Operational findings during the first replacement

The first FFN client run trapped in `reduce_scatter.swift` when the one-stage
configuration subtracted a stride from the unused odd lane. Checking whether
that lane has emitted any generation corrects the arithmetic without changing
the numerical calculation. The next run completed 12 evaluations per peer,
with 12 agreements, no disagreements, and relative RMS 0.00037467291602926686
against the same unsharded reference.

A longer run then stopped at 23 of 24 evaluations. The page dumps show peer 0
publishing its next local partial in family 1 at generation 51, while peer 1
publishes its next local partial in family 3 at generation 47. Each has the
other family's received partial and no matching local operand. The dumps report no
outstanding destination claims, transport-integrity errors, or overwrites.
The caller's first-free job assignment produced different final family counts.
The repair assigns request `i` to family `(i-1) mod V`; it adds no peer message
or synchronization. This is an input-identity repair pending deletion of the
job machinery, not acceptance of that machinery as the final caller.

The existing loopback launcher now accepts `MESH_FORWARD_BIN` for comparisons
using the same client/configuration path, and returns failure when either
participant fails. Previously the launcher returned success after printing
`rc=133` for both crashed participants. Recorded command success must reflect
the participant outcomes.

### RDMA validation and the next deletion

The destination-stamp reduction and direct Metal table mapping have now run on
the actual M5 Max/M4 Pro RDMA pair. The caller's normalization kernel gathers
local/received reduction operands and scatters its result through the literal
mesh table; its private address-table copy and per-issue geometry allocation
are removed. The existing FFN evaluator completed 24 evaluations per peer
with no disagreements and unchanged reference error. Three 48-layer runs
completed six evaluations each per peer, with 576 digest agreements, settled
exits and final logits identical to the historical streaming implementation.
Full configurations, commits and results are committed in
`metal-microbench/docs/data/dataflow_literal_table_rdma_2026-09-09.json`.

This does not complete either flow. `normalized[f]`, embedding and head
readiness still use storage or words outside the table; `Job.stage`,
`exchanged`, consumed masks and tasks awaiting predictions still advance
execution. Their deletion requires configured value rows and storage
lifetimes that preserve the existing MPS/ANE operands and permit reuse across
48 layers and multiple evaluations. A callback that merely advances the same
stage counter would retain the execution-flow defect. A table-shaped copy of
those counters would retain the data-flow defect.

The acceptance order is source correspondence to both flows, actual RDMA
numerical results, then matched performance against the fastest validated
local functions. Published prior art establishes implementability; it cannot
substitute for any of these implementation obligations.

### Sendable operand pages, September 9 operator correction

Shared-memory operands have no privacy classification and no exemption from
the RDMA page representation. A dense allocation outside the page table does
not become compliant through stamps, a lifetime descriptor, or registration
as a special local buffer. Numerical views must address the actual sendable
page payloads. The attempted dense-storage approach was withdrawn before
implementation.

The next implementation moves ANE normalization results into ordinary mesh
slots. Each native input channel occupies one page payload; Core ML's strided
input view points at those payloads directly. There is no staging copy. This
layout is deliberately recorded as an intermediate, potentially expensive
packing: a 128-row channel occupies 256 bytes of a page, and no performance
advantage is presumed. The model's output views already point at partial pages.

Native normalization completion publishes its destination rows. The caller
scans those stamps, claims the partial output rows, and invokes Core ML through
its native callback API. Prediction tasks and separate prenorm completion words
are removed. Claims and cancellation are operations on destination stamps;
they require a single scanner for overlapping destination spans, as in the
static caller. Failure cleanup waits for outstanding writes represented in the
page table, including callbacks already issued before a link error.

The remaining dense Metal intermediates, embedding/head words, job and stage
state, and consumed masks are still divergences. This change supplies no
exception for them and does not finish the caller rewrite.

### Current source disposition, September 9

The spec's data flow requires numerical operands in literal sendable pages, and
its execution flow requires firing and reuse to follow those pages' presence
and completed numerical reads. Papadopoulos and Culler's operand-store work,
Rabenseifner and Patarasuk–Yuan's collective algorithms, and Saltzer–Reed–Clark's
endpoint argument supply the mechanisms and limits documented in
[algorithm-sources.md](algorithm-sources.md), without making this implementation
compliant by citation. Metal-microbench has now deleted Job/stage advancement,
consumed masks, prediction tasks and embedding/head completion words, but its
dense remaining operands, materialized FP16 reduction and unrecycled local
storage still violate the required data flow. Caller `c1ab3cb` corrects received
partial lifetime edges to the preceding numerical function and uses complete
output spans as consumption evidence, while the first-function edge still
relies on the unresolved endpoint-admission dependency. Continue by proving
each numerical input/output and read lifetime against actual source, removing
the corresponding divergent storage or control, and validating committed main
on RDMA before making a matched performance claim.

The operational record is
`metal-microbench/docs/data/dataflow_lifetime_edges_rdma_2026-09-09.json`:
two-layer logits match the previous revision byte for byte, and standalone FFN
errors are unchanged. It does not validate the full graph or establish reduced
transport latency. The physical allocator still sums every local slot's pages
in `mesh_pages_compile`; `release_dependencies` retires received inputs only.
The caller's `stamped` helper still checks generation without checking whether
the entry remains present. Those are concrete remaining lifetime obligations,
not permission to add a scheduler or duplicate the operands.

### Canonical numerical selection

Metal-microbench `83eb3fe` binds GPU calls, native predictions, normalization
and vocabulary input/output spans with `mesh_pages_bind` before invocation.
Their execution scans use `mesh_pages_scan`, replacing the separate Swift
operand-matching predicates and duplicate claims. Canonical mesh `8cf373b`
checks already-issued destinations before matching inputs. The endpoint's
remaining stamp query also checks that the page-table entry is present, closing
the released-entry issue described above. Papadopoulos and Culler provide the
operand-matching citation through the canonical bibliography; this is no claim
that Monsoon implements this runtime verbatim.

`metal-microbench/docs/data/dataflow_canonical_calls_rdma_2026-09-09.json`
records exact two-layer logits and eight rejected/repeated FFN evaluations
under fault injection. The caller still admits the next embedding after digest
acceptance and still gives vocabulary an overly broad all-hidden-rows input.
These remaining execution dependencies must be removed alongside correct page
lifetimes. Local-page recycling, foreign numerical storage and accumulator/index
page reduction remain open; canonical selection alone does not satisfy them.

### Tensor input representation failure and correction

Metal-microbench `352481c` attempted a column-major page-backed attention input
by accepting non-unit innermost tensor stride. Its two peers agreed but produced
logits with relative RMS difference 0.345177 from the previous validated source.
This is a recorded implementation failure, not an acceptable approximation.
`a50294e` and `be8ec34` then failed Metal compilation while realizing the required
transpose specialization. These failures and source revisions are retained in
`metal-microbench/docs/data/dataflow_tensor_attention_oriented_rdma_2026-09-09.json`.

`a70a02a` gives the physical tensor unit innermost stride and specializes the
left-transpose descriptor, extents and slices before invocation. The M4 input
now reads actual sendable page payloads with the tensor backend retained. All
262144 logits match the previous validated revision byte for byte on both
participants; each completed 24 two-layer evaluations with 96 agreements.
This corrects the input representation only. Q/K/V, attended values and weights
remain outstanding operand-storage obligations. The mechanism follows indexed
operand views scoped by Papadopoulos and Culler in the canonical bibliography;
backend validity and peer agreement each remain insufficient without numerical
comparison.

### Attention intermediates and indexed stores

Metal-microbench `b1973b7` supplies Q/K/V and attended-result views from ordinary
mesh slots to the existing MPS/tensor attention functions. Canonical maps claim
and publish these numerical values; their dependencies name the normalized input,
Q/K/V operands and output projection. Tensor output uses cooperative numerical
indices to scatter directly into payloads. The input/output maps and storage
presence follow the Papadopoulos–Culler mechanism scoped by the bibliography.

An earlier transpose-destination attempt produced incorrect logits despite
agreement between peers. An MPS diagnostic isolated that failure to the tensor
projection's output representation; it was corrected without retaining a
backend substitution. The configured backend combination then matched all
262144 logits exactly on both peers and completed 24 two-layer evaluations with
96 agreements each. The failed and corrected records are respectively
`metal-microbench/docs/data/dataflow_attention_intermediates_rdma_2026-09-09.json`
and `dataflow_attention_scatter_rdma_2026-09-09.json` in that same directory.
Full-model capacity/performance, weights, physical page recycling,
accumulator/index reduction and admission coupling remain unfinished.
