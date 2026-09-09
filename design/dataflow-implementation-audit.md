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
