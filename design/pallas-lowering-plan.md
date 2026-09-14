# Nine steps to Pallas-style lowering

Plan based on source `70b3703`, September 13, 2026. This is an implementation
plan, not a claim that the lowering or its performance criteria are complete.

## Target and constraints

Write numerical functions over values, indices and masks; realize their layout,
placement, storage and executable functions once; repeatedly execute the realized
program with independent producer and consumer progress. Changing the physical
lowering must preserve that program's numerical and publication contracts.

No extra steady-state interpreter, Python numerical callback, graph construction,
operand allocation, tensor-wide rendezvous or software transfer acknowledgement
is introduced by compilation. Canonical registered pages remain the shared-memory
operands. Local register/threadgroup scratch is a numerical implementation detail,
not a second host operand store. Existing page stamps remain readiness; compiler
representations are setup data and do not become a second participant scheduler.

No-overhead and no-feature-regression are acceptance requirements. They are not
assumed consequences of adopting Pallas-like syntax. Performance must be evaluated
against the fastest validated implementation of the same work and precision on
these machines, including first-result latency and sustained throughput.

## Published mechanisms to reuse

The JAX authors' [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)
separates kernel bodies, references, index maps, grids and backend lowering. That
is the structural prior art for steps 2–4. Its discussion of batching supports
transforming a grid and its index maps instead of executing a Python batch loop.

The [Pallas pipelining derivation](https://docs.jax.dev/en/latest/pallas/pipelining.html)
explains buffer reuse, independent transfer/compute work and retaining local
accumulators across reduction iterations. Steps 5–6 apply these mechanisms while
preserving independently usable region publications. Publication does not imply
a kernel boundary, completion rendezvous or pause in the publishing computation.

The [Pallas collective matmul](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)
reuses an optimized local matmul, overlaps communication with persistent compute,
and uses distinct receive storage to avoid extra backpressure communication.
Steps 4, 6 and 7 adopt those principles through mesh's existing bridge. TPU/GPU
synchronization primitives are not transplanted into application control flow.

Apple's [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
is authoritative for the actual SEND/RECV and frame-credit transport. Pallas is
prior art for lowering and composition, not evidence that this link supports
remote-write hardware or GPU-originated verbs.

## 1. Record the semantic and performance contract

Inventory the operations, dtypes, shape/stride behavior, peers, numerical errors
and derivative operations already supported by `python/mesh` and
`xonotic/solver/strat/tensor*.py`. Include Xonotic expert and neighborhood VJPs,
ragged tails, broadcasting, transposition and aliasing. Read actual implementations;
do not equate a symbol's existence with backend coverage.

Record externally usable output regions and their true dependencies. The required
properties are independent input consumption, independent output publication,
FP32 contraction/reduction storage where specified, exact endpoint identity,
correct source lifetime and bounded repeated invocation storage. Normalization
still requires its own row statistic; unrelated rows do not become dependencies.

Use the existing gold, streaming-overlap and Xonotic numerical workflows to record
matched baselines before replacing each path. Keep setup time separate. Existing
reports are evidence for their recorded shape/dtype only.

Done when every existing numerical path has a named replacement owner, an
observable contract and an applicable baseline or an explicitly missing baseline.
Missing evidence is work to do, not a reason to delete the behavior.

## 2. Extend one typed numerical representation

Extend `python/mesh/kernels.py`'s expression machinery and the existing
`Program.kernel_call` entry point. Represent logical rank, shape, dtype, strides,
index expressions, masks, loads, stores, arithmetic, contractions, reductions and
multiple outputs. Add comparisons/selects, gather and scatter-add with explicit
collision semantics. Retain output and accumulator precision separately.

Keep logical rank independent of the native two-dimensional matrix view. N-D
indexing lowers to canonical page views and supported native matrix layouts;
flattening must retain strides and axis identity. Ragged masks describe numerical
validity, not whether an entire function is allowed to make progress.

A setup trace builds this representation once. Reuse the existing native operation
bindings and code-generation facilities. One public kernel body must not require
separate handwritten CPU, Metal and model-specific streaming definitions.

Done when the existing pointwise, contraction, gather and reduction bodies are
expressible through this representation, with typed index/mask and multi-output
examples, without adding another public execution interface.

## 3. Lower index expressions to region dependencies

Extend `BlockSpec`/view resolution and setup binding to retain logical-to-physical
maps across existing blocks. Affine indexing yields static read/write regions;
masked accesses retain validity domains. Independent stores yield independently
publishable outputs. Keep both source and target index vectors throughout.

Dynamic indexing is not statically knowable in general. For expert routing and
neighborhood lookup, treat the routing indices as ordinary numerical inputs:
index computation produces explicit indices into setup-allocated storage, and
indexed consumers use those indices with canonical page readiness. Retain the
actual indices rather than guessing their range or rebuilding a graph on the host.
The concrete representation must fit the existing page-table specification; any
needed extension is documented before implementation.

Scatter-add must identify when a destination region's contributions are complete.
Use explicit contribution domains and FP32 partial values/reductions; a partial
write must never be mistaken for the completed sum. Duplicate indices, empty
contribution sets and overlapping outputs need defined algebra. No global all-input
wait is introduced merely because one destination has multiple contributors.

Done when indexed programs complete independent output regions with an unrelated
index/data region absent, including duplicate-index reduction and ragged cases.
Source inspection must establish the dependency sets, not infer them from timings.

## 4. Lower to existing efficient backend functions

Bind recognized contractions to the existing CPU/MPS/native paths during setup.
Lower the remaining numerical expressions to compiled CPU/Metal code through the
same interface. Preserve externally supplied numerical kernels with explicit
region bindings; remove accidental Python execution from compiled paths, without
removing the capability those callbacks supplied.

Resolve backend, precision, layout and kernel specialization before invocation.
Use measured shape curves to select efficient kernels. Address CPU FP16's current
scalar path with a validated numerical implementation. Core ML precision, device
placement and internal-copy evidence remain separate questions; matching output
backings alone proves neither ANE execution nor internal zero-copy behavior.

Done when numerical equivalence and region publication hold on both available
SoCs and no path is switched to a slower backend for binding convenience.

## 5. Fuse work without erasing observable progress

Perform setup-time expression simplification, common-expression reuse and legal
fusion. Combine pointwise epilogues and reductions over ready local contributions;
retain FP32 arithmetic and specified casts. Avoid compulsory shared-memory storage
for every compiler-internal scalar or partial value.

A remotely consumed or otherwise independently observable output must remain
available to its readers as soon as its required writes are visible. This is an
asynchronous publication obligation, not a requirement to split kernels, finish
the enclosing function, or await delivery or consumption. Composition must
preserve publication and fanout while other work continues. A missing consumer
operand must not delay the producer's computation or publication. Do not retain
a published value only in scratch inaccessible to its configured readers.
The operator's [transcript contract](SPECIFICATION.md#25-asynchronous-concurrent-publication-current-mesh-session)
governs this distinction; existing launch boundaries do not define semantics.

Issue already-ready regions efficiently through the existing execution owner.
Any grouped command submission must preserve the required completion/publication
granularity; grouping is not permission to delay early outputs until the slowest
operation finishes. Persistent execution is a backend implementation option only
where its publication and memory-ordering mechanism is established.

Done when source demonstrates fewer unnecessary launches/stores and traces show
that the removed overhead was not exchanged for delayed producer emission or
consumer start.

## 6. Realize storage lifetimes and pipeline depth

Plan canonical operand/output storage, local scratch and simultaneous value
instances before execution. Elide allocations only for internal values whose
uses the lowering actually subsumes. Preserve distinct destination buffers for
simultaneously live outputs and explicit source holds until all readers and
transport uses complete.

Reuse storage only when existing lifetime facts prove it legal. Do not add a
new dependency just to reuse fewer pages. Repeated invocation uses bounded
preallocated capacity and canonical readiness, not an ever-growing cloned graph
or a second application scheduler. Keep the visible memory preflight and useful
byte lengths; report reserved, registered and live operand bytes separately.

Done when repeated concurrent invocation has stable storage, survives delayed
consumers and fanout, and preserves early progress without allocating during the
numerical call graph.

## 7. Compose placement and collectives from the same operations

Represent participant placement and all-gather, reduce-scatter, all-reduce and
indexed redistribution as region algebra with explicit peers. Realize routes,
receive destinations and local reduction functions during setup. Transport sees
indices and lengths, never FFNs, expert models or graph interpretation.

Keep the publication-fed bridge and explicit transfer tuples. Complete per-QP
accounting using retained actual frame costs, queried capacities and adequate CQ
storage, so shorter payloads recover concurrency safely. Preserve posted-request
identity and hardware ordering obligations without blocking unrelated ready work.

Describe logical batching as grid/index-map transformation. Preserve current
explicit derivative operations when transforming their regions; broader automatic
differentiation is a subsequent capability, not grounds to drop existing VJPs.

Done when the same bodies compose local and distributed executions, and fanout,
reduction and redistribution retain streaming across available participants.
More-than-two-node performance claims require that hardware evidence; a two-node
run does not certify arbitrary topology scaling.

## 8. Migrate callers and remove duplicate lowering

Move `nn.linear`, FFN, RMSNorm and embeddings onto the shared representation.
Migrate Xonotic's expert/neighborhood bodies and derivatives using steps 2–3,
then remove their whole-region `grid=(1,)` binding where the operation permits
independent regions. Preserve genuine full-reduction dependencies where required.

Delete replaced model-specific streaming builders, duplicate emitters and
compatibility interfaces as their complete replacements land. Preserve supported
numerical behavior, not obsolete implementation topology. Compiler code belongs
to the library; transport and invocation control stay in canonical mesh.

Done when ordinary callers express mathematics and placement, and none needs to
hand-build partial-sum launches, inline a model in transport, or repair missing
metadata with control flow. Every path inventoried in step 1 must be accounted for.

## 9. Establish the end-to-end result

Extend existing operational examples and Xonotic workflows, without adding a
parallel evaluator. Cover the FFN gold, indexed expert/neighborhood composition,
ragged regions, duplicate scatter indices, fanout, multiple live invocations and
repeated buffer reuse. Retain exact cancellation/overflow evidence and existing
numerical precision requirements.

Inspect the compiled functions, retained dependencies, bindings and pointer flow
for hidden staging, host numerical callbacks, runtime allocation/compilation,
whole-operand barriers and lost publication boundaries. Retain actual index and
output identities in diagnostics so the evidence does not reconstruct discarded
metadata.

Measure first usable output, consumer start, sustained throughput, launch count,
bytes moved, memory footprint and computation overlapping outstanding transport.
Do not equate host command intervals with physical GPU/wire occupancy. Use the
fastest validated local numerical baseline for the same shape, precision and
composition, then report additional distributed gain and compounded gain separately.
Every percentage uses online count, mean and sample variance. Preserve raw data
and source revisions; timing uncertainty cannot establish no regression.

Done when the feature inventory is covered, source establishes the structural
invariants, and matched measurements show no unresolved latency/throughput or
memory regression in the covered workloads. No universal zero-overhead or
hardware-independent peak-performance claim follows from that finite evidence.

## Implementation order and first deliverable

Step 1 starts immediately before each replacement. Implement steps 2–4 as one
vertical slice: an indexed gather → pointwise transform → scatter-add program,
with duplicate indices and an unrelated delayed region, through the existing
kernel_call and page-table execution. Include the corresponding Xonotic operation
as a caller so the representation is not designed around a standalone sketch.

Then implement steps 5–6 on that same representation, step 7 for collective
composition/capacity, and finish migration in step 8. Apply step 9 throughout;
its final audit closes the full implementation. Commit and push each complete
source increment to main and the peer before distributed measurements. Preserve
old source only in Git history. This plan itself does not authorize claiming those
future implementation steps complete.
