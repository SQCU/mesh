# Complete dataflow replacement: requirements and acceptance

This is the consolidated implementation contract for the September 9, 2026
instruction to enumerate the remaining work, implement it together, run real
inputs, and review source and execution until complete. User instructions govern;
this document does not authorize exceptions. It supersedes completion implications
in the chronological helper/evaluation notes. Those notes remain historical evidence.

Initial source review: mesh `97d87e4`, metal-microbench `d177586`, both on `main`.
The active numerical caller is `runReduceScatter` in `reduce_scatter.swift`.
It calls `mesh_pages_*`. The `mesh_rows_*` replacement is not used by that caller.
Compiling both runtimes into one executable does not constitute migration.

## Allowed representation and execution

Configuration realizes a finite feed-forward graph before invocation: numerical
functions, their input/output row maps, participant ownership, tensor geometry,
backend specializations, page allocations and lifetimes, transport bindings,
and output/metadata return maps. Configuration may validate these contracts;
numerical invocation must not reconstruct or repair them.

Runtime values occupy the actual registered page payloads. The table records
page identity, stamp and use count. Accumulators, indices, metadata, parameters,
constants and intermediate tensors are values too. A Metal or CoreML view may
alias those pages; a descriptor for a separate allocation does not make that
allocation compliant. Registering an address span may cover many pages; this
does not authorize a different operand buffer or a larger logical transport unit.

The one caller scans the configured functions and input/output stamps. It issues
the rows available at that scan in one command buffer per function, and completion
publishes the corresponding rows. Native numerical API invocations need the same
publication contract without a second scheduler. Dependencies inherent in the
algebra remain: attention cannot read a key/value row that is absent, and two
writes cannot concurrently mutate the same accumulator without a defined reduction.
These dependencies are represented by values and rows, never a job, stage, phase,
completion token, digest verdict, prediction task, or mirrored readiness counter.

Provider work requests and completion records must refer to literal pages. They
are not an additional application readiness protocol. Submission capacity is a
physical constraint, not permission for sleeps, spinning inside a numerical
function, retries of failed computation, or a second queue of numerical work.
Every pending action must remain discoverable from its configured rows.

`out, meta = meshfunction(x)` returns configured row references without waiting.
Metadata carries literal error codes, their domain, and where/when they arose.
Composition carries metadata alongside numerical values without inspecting errors
to control execution. Only the outer calling context interprets errors, digests,
missing results and deadlines and may repeat the entire NFE. A failed operation
must not fabricate successful numerical output stamps.

## Requirements matrix

“Open” means completion evidence is absent, including where partial code exists.
Each row requires source review of its entire dependency path and the stated
runtime evidence. Passing another row is not a substitute.

### Memory budget established by the operator

The M4 Mini reports `hw.memsize = 25769803776`: 24 GiB of unified memory.
The operator instructed that allocation below 80% should be treated as available
for planning. Use **19.2 GiB** for its complete allocation plan, rather than
mistaking the existing bridge allocation for the machine's capacity.

The observed bridge runs with `-M 25`, allocating 6 GiB: 393216 receive pages
and 1179648 transmit pages, each 4096 bytes. Its 4.5 GiB transmit arena is a
configuration choice. With the current one-quarter receive/three-quarter transmit
split, an 80% bridge allocation would provide approximately 4.8 GiB receive and
14.4 GiB transmit storage. Physical-page rounding and header/descriptor/table
storage must be included in the final budget; 19.2 GiB is not a claim that all
of that space is numerical payload.

For the existing M4 ownership, the 48 FFNs require
`48 × 3 × 3456 × 3840 × 2 = 3822059520` bytes of FP16 weights.
The tied embedding/head requires
`262144 × 3840 × 2 = 2013265920` bytes, counted once.
Their combined 5.4345703125 GiB excludes attention weights, all intermediate
values, indices, metadata and page padding. This exceeds the old arena but fits
within the operator's revised planning budget; it is not a hardware-capacity
blocker. Finish C05 with the complete graph and actual backend layouts.

Do not preserve the old separate dense/native allocations while also allocating
page-backed copies. Configure the complete storage ownership first, then realize
the corresponding bridge allocation and numerical bindings. The existing
`mesh_pct + app_pct` wire-limit calculation also needs to follow that ownership;
it must not count the same shared numerical pages twice.

The bridge remained paired and responsive with no attached computation when
these values were read. No bridge restart or allocation change was made by this
budget update. The last source synchronization built both callers successfully
at mesh `97229a1` and metal `e988499`; those were documentation-only changes,
and neither build result establishes replacement-runtime execution.

| ID | Required result | Current disposition and acceptance evidence |
|---|---|---|
| C01 | One configured graph and one mesh numerical caller | Open: migrate `runReduceScatter`; trace serving and benchmark entry points to that implementation. No parallel replacement evaluator/caller. |
| C02 | Configuration precedes invocation | Open: bind every kernel, matrix, CoreML model, layout, allocation and lifetime before launch; inspect the transitive numerical path for allocation, compilation, backend inference and parsing. |
| C03 | Every logical read has a producer and exact use multiplicity | Partial: `mesh_rows_realize` checks source/use coverage. Add complete-graph accounting for parameters, shared constants, transport reads, caller returns and metadata. No double releases. |
| C04 | Static graph is feed-forward and geometrically valid | Open: prove graph dependencies, bounds, element types, padding, aliases, repeated maps and per-invocation identities during realization. No runtime repair branches. |
| C05 | All graph storage fits the configured physical capacity | Open: produce a complete allocation/lifetime ledger for 48 layers and one/two simultaneous NFEs. Current replacement rejects every physical overlap, even disjoint lifetimes. |
| C06 | Safe reuse follows actual dependent reads | Open: represent reuse through page ownership/use counts and graph lifetimes, including NIC source reads and callbacks. No separate reuse gate or admission verdict. |
| C07 | No dense/private operand exceptions | Open: replace active dense weights, norm/gamma/scale, embeddings, tokens, attention and FFN intermediates with actual page payloads. Report backend-owned storage as unresolved until its binding is demonstrated. |
| C08 | Load parameters into their final pages | Partial: `ModelFile.loadRows` exists but is unused by the caller. Exercise actual model tensors, tails, transposition, BF16/F16/F32 conversion and reloading after invalidation. |
| C09 | Preserve winning SoC numerical functions | Partial: explicit weight views now reach FFN, attention and vocabulary Metal bindings; head normalization honors weight byte offsets. Existing dense results prove arithmetic only. Complete actual page-layout binding on each configured backend; do not silently replace ANE/MPS with a slower convenience backend. |
| C10 | CoreML storage contract covers the whole operation | Partial: both machines wrap a literal mesh span with IOSurface at the same address. The actual M4 FFN uses MLE5Engine; its local 128-row input and two outputs now report direct binding through IOSurface-backed MLMultiArray, with byte-identical output versus the former copy-bound layout. See `algorithm-sources.md#native-binding-observation`. Bind complete mesh layouts, weights, intermediates and completion; local I/O proof does not close the whole operation. |
| C11 | All tensor views respect payload boundaries | Partial: `MatrixView` supports paged axes; validate actual page-table-backed views, receive-page gathers, MPS subviews, tensor tiles and nonaligned contraction tails without reading headers as numbers. |
| C12 | Readiness lives only in the table | Open: replace old bitmaps, changed/runnable queues, status gates and caller admission state. Review select/claim/complete behavior for arbitrary ready subsets and repeated invocations. |
| C13 | Issued indices retain their value lifetime | Open: selected indices are configured page values; completion must publish precisely the issued rows even when subsequent scans issue others. No per-call heap arrays or per-job records. |
| C14 | One submission covers available rows per function scan | Open: remove fixed row-group command-buffer fan-out. Preserve genuine kernel shape constraints as configuration, not an application scheduler. Count command buffers against actual ready subsets. |
| C15 | Independent computation, reduction and transport can proceed | Open: no queue-wide waits, dependency-free serialization, or awaiting predictions. Measure ready-to-submit intervals and overlapping device/transport spans under real inputs. |
| C16 | Attention dependencies are literal | Open: expose Q/K/V, attention and output contraction values with their real full-context or mask dependencies. No hidden scratch or invented partial-attention readiness. |
| C17 | Distributed arithmetic matches the model algebra | Open: partition contraction ownership, scatter partials to owners, reduce in FP32, gather the reduced values and apply RMS/residual/scale in the specified order. No early FP16 materialization of an FP32 accumulator. |
| C18 | Reduction is incrementally usable and race-free | Partial: `row_add` adds a matching set of partial pages into its output. This agrees with the spec's pairwise operation; overwriting that output is not a defect. Incrementality is across ready page pairs, not repeated updates to the same accumulator page. Integrate one writer per accumulator value and static contribution/index dependencies. |
| C19 | Every reduction input stays within its page | Partial: addition now consumes configured input page maps and addresses each payload separately, including FP16 expansion and tails. `mesh_rows_validate_add` validates exact page counts before launch. Active-caller integration and real-input evidence remain open. |
| C20 | Publication has one owner | Partial: CPU addition now writes values only; generic `mesh_rows_publish` owns all output stamps and input releases. Each addition has a configured index-page output. Active caller, GPU and native completion integration remain open. |
| C21 | Actual RDMA pages carry the graph | Open: replacement send/receive functions lack an integrated owner in the active caller. Exercise real M5/M4 SEND/RECV, returned receive pages and transmit source lifetimes without copies or loopback. |
| C22 | Transport follows actual provider capabilities | Existing UC SEND/RECV matches Apple TN3205; no RDMA_WRITE implementation is supported here. Configure the discovered page geometry, registration and WR limits before invocation; remove hardcoded geometry inconsistencies. |
| C23 | Transport completion is not numerical admission | Open: CQ completion may publish a received page and release a NIC read. Remove application FIN/OPEN/READY/abort handshakes and digest/error gates after migrating their callers. |
| C24 | No recovery in computation/transport/reduction | Open: old stream retries, abort/recovery protocols and numerical status cancellation remain. Separate fleet provisioning/reachability from NFE execution; do not remove the operational protections against wedging the verbs driver. |
| C25 | Nonblocking metadata preserves original provenance | Open: `mesh_rows_report` exists; bridge `L_FAULT` collapses original errors into `bad`/link-down, and Metal/CoreML map errors to EIO. Preserve source domain, literal code, operation, row/function, participant and occurrence. |
| C26 | Metadata capacity and composition are configured | Open: allocate occurrences and graph return maps before launch, including submission and completion errors. Define global link errors without falsely assigning them to a successful numerical operation. No numerical dependence on metadata stamps. |
| C27 | Native failures do not gate the graph | Open: CoreML callback error returns and `backingUsed` assertion, Metal status checks and `mesh_pages_cancel` remain. Successful physical completion publishes outputs; errors publish metadata. Missing output remains absent without an error-consumption branch in downstream functions. |
| C28 | Invalidation is exposed directly to the outer caller | Open: no replacement invalidation API exists. Detach the old table from new invocation use without a synchronous drain/recovery operation. |
| C29 | Late accesses cannot corrupt a replacement table | Open: current receive ignores the address epoch; ACK uses the submitted page's mutable address. Establish transport/table identity and GPU/NIC memory ownership across invalidation before reuse. Merely clearing stamps is insufficient. |
| C30 | Hung work cannot require a computation-layer wait | Open: identify the backend/OS mechanism that ends access, or retain its old physical ownership outside the new graph until access ends. Do not promise immediate physical reuse while an uncancelled device can still write the address. |
| C31 | Caller alone restores parameters and repeats NFE | Open: honor the NVMe/transitive-NVMe deployment invariant; exercise caller-owned whole-graph repetition and reload, never parity, partial retry or residency probing inside the graph. |
| C32 | Diagnostics do not enter the numerical dependency path | Open: remove inline logging, clock sampling, status inference and benchmark rendezvous from numerical functions. Collect timing as asynchronous metadata or external observation with overhead reported. |
| C33 | Real input reaches the same caller used for serving | Partial at numerical main `8f02bf4`: `runReduceScatter` accepts explicit token IDs, both entry modes use its one call site, and the existing runner records the supplied IDs. Both builds passed. Replacement/serving integration and complete real-input execution remain open. |
| C34 | Numerical evidence covers the complete NFE | Open for replacement: real model and input identity, all 48 layers, both peers, one/two simultaneous NFEs, finite output and appropriate reference/logit metrics. Old-caller agreement does not establish this. |
| C35 | Errors and invalidation are externally reviewable | Open: exercise the existing operational/client path through an actual error and caller-selected rerun; show literal metadata, unchanged numerical-channel semantics and no stale writes into new outputs. No mock control-flow suite. |
| C36 | Performance uses the fastest validated local baseline | Open: record the same model/shape/input/backend baseline for each SoC, local improvement separately, extra TP gain, and compounded end-to-end gain. No comparison against a binding-induced slow baseline. |
| C37 | The 10 microsecond question is measured | Open: report distributions/maxima for host scan, launch, ready-to-issue, transport residence, completion-to-publication and reduction, plus measurement overhead. A percentile is not a universal upper bound; report every observed exceedance and attribution. |
| C38 | Communication cost is accounted for | Open: count useful bytes and actual per-page/header/hop traffic, command buffers and WRs; compare with the collective cost model. Do not attribute a one-second NFE to wire latency without a timeline. |
| C39 | Whole-source deletion follows caller migration | Open: remove excluded runtime definitions, callers, build/header dependencies and obsolete configuration only after C01–C31 are integrated. Preserve supported non-NFE functionality or migrate its caller first. |
| C40 | Reproducible source and deployment | Established workflow: commit on main, push/check out main on every machine, verify identical heads, build all participants concurrently. Extend builds to the bridge when changed; record running bridge executable identity, not just repository HEAD. |
| C41 | No parallel or weakened validation | Use existing numerical/client evaluators and operational measurements. Retained focused checks must cover a failure client metrics miss; do not add tests that mirror implementation topology. |
| C42 | Source and documentation remain maintainable | Every added function has a documentation citation to the listed literature; source has no prose comments. Count maintained serving dependencies, helpers, generators and tests honestly; separate documentation migration from structural reduction. |
| C43 | Completion requires a closed evidence matrix | Open: review the entire active source path and runtime trace after the integrated implementation; repair all remaining violations and repeat. An inventory, a build, an unused helper, or an old-caller regression cannot close the task. |
| C44 | Immutable parameters work with simultaneous invocation stamps | Open: define shared parameter row identity and lifetime for different in-flight stamps without copying parameter storage or changing a row stamp while another invocation depends on it. The current exact-stamp matcher and disjoint-physical-output rule do not establish this. |

## One integrated implementation, then removal

The implementation must be reviewed as the following connected change, not as a
sequence of independently declared successes:

1. Realize complete graph row maps, lifetimes, metadata occurrences and backend
   views, including parameters and the caller's input/output boundary. Resolve
   native storage and invalidation ownership before adopting an ABI that cannot
   express them.
2. Implement the required page functions with one publication contract:
   configure, match rows, encode arithmetic/gathers/scatters, publish values,
   accumulate contributions, publish indices/metadata, transmit, receive,
   release and invalidate. Function names alone do not establish those semantics.
3. Replace the body and bindings of the one `runReduceScatter` caller. Connect
   native and Metal completion and actual bridge metadata. Keep the old code only
   until all active callers have migrated; do not choose runtimes during invocation.
4. Delete excluded code and build dependencies, then build identical committed
   sources on both machines. Run the existing client evaluation on real inputs,
   full graph and actual RDMA. Review every requirement against the actual binary
   and trace, fix the remaining gaps together, and rerun affected measurements.

## Whole-change source review, September 9 continuation

Operator correction: isolated helper patches followed by builds and old-runtime
regressions are not the requested workflow. The complete implementation is the
unit of revision and evaluation. The completed build at numerical `8f02bf4` and
mesh `952ae2f` is a build result only. Do not start another partial NFE regression
as a substitute for the connected replacement described above. Source may be
preserved in main commits while the connected edit is unfinished; that does not
make such a revision an accepted implementation or a performance baseline.

### Complete numerical value graph

For each attention or FFN function l, let X_l be the full hidden input, P_l^r
the contraction partial computed by participant r, and O_r the configured output
row ownership. For the current two-participant realization, O_0 and O_1 partition
the input rows. Weight ownership partitions the contracted dimension, independently
of output row ownership. The following is the single connected graph to realize:

| Numerical function | Literal inputs | Literal outputs | Execution dependency |
|---|---|---|---|
| Embedding | Caller token pages, tied embedding parameter pages | X_0 pages | Actual token and parameter rows |
| Pre-normalization | X_l pages, pre-normalization parameters | Normalized input pages in the chosen local backend layout | Complete hidden dimension of each numerical row |
| Attention projections | Normalized input, Q/K/V and normalization/rotary parameters | Q/K/V pages | Configured projection rows |
| Attention | Q/K/V pages, configured mask/position values | Attended pages | Actual context/mask inputs of the existing winning function |
| Attention output contraction | Attended pages and owned output weights | P_l^r pages | Existing contraction's numerical inputs |
| FFN gate/up and activation | Normalized input and owned gate/up weights | Up and activation pages | Existing fused or unfused numerical composition |
| FFN down contraction | Activation and owned down weights | P_l^r pages | Existing down contraction's numerical inputs |
| Scatter | P_l^r restricted to O_peer | Received partial pages at the owner | Each produced partial page independently |
| Pairwise reduction | Local and received partial pages restricted to O_r | FP32 accumulator pages and index values | Matching partials for the same elements; one writer per output |
| Owner epilogue | Complete FP32 accumulated hidden row, gamma, X_l residual and scale | X_(l+1) restricted to O_r | All contributions to that hidden row; RMS then residual/scale in model order |
| Gather | Owner epilogue pages | Received X_(l+1) restricted to O_peer | Each completed owner output page independently |
| Final normalization and head | Last X pages, output normalization and tied embedding weights | Caller logit pages | Actual head input rows and contraction dependencies |
| Digest/compare | Retained partial/output pages and received digest values | Digest and error-metadata pages | Its own value dependencies; no edge back into numerical admission |

This is Rabenseifner/Patarasuk–Yuan's partition, reduce and gather applied to the
existing model algebra; Papadopoulos–Culler's named-value firing supplies execution
readiness. It is not a new collective or a second implementation of a local kernel.
For more than two peers, the configured reduction/exchange graph supplies the
additional pairwise functions; the numerical caller must not interpret topology.

The owner epilogue consumes FP32 accumulator values directly. The current active
path's materialized half-precision reduction followed by normalization on both
participants is not this value graph. The replacement must not preserve that
intermediate FP16 rounding or duplicate the owner epilogue for binding convenience.
Attention and FFN remain distinct numerical compositions with their actual
storage/dependencies; neither becomes a hidden monolithic scheduling callback.

Native-contiguous input storage need not be the receive landing layout. The
required pre-normalization computes directly from local/received X pages into
its final native input pages. This is required arithmetic, not a packing copy.
The native output, weights and internal intermediates must also have literal
page ownership; fixing just the native input does not satisfy C07/C10.

### Complete ownership and completion contract

One allocation ledger must cover every value above, all 48 layers, each supported
in-flight invocation, backend padding, aliases, receive capacity, issued-index
pages, metadata occurrences and retained storage after invalidation. It must
identify which addresses refer to the same physical pages. Native/Metal wrappers
and registered spans are views of that ledger, not additional operands.

Immutable parameter values may have distinct invocation row identities pointing
to the same physical backing. Realization must account for every such read and
retain physical ownership until all corresponding uses end; stamping a shared
logical row with a second invocation's stamp is incorrect. Mutable values may
share physical storage only with a demonstrated lifetime ordering. The current
replacement's blanket rejection of physical overlaps is not a realization of
these lifetimes and cannot be copied into the final allocator as its contract.

Every submission's selected indices are themselves retained page values. The
completion publishes exactly those rows, even if a later scan has issued more
rows for the same function and stamp. Scanning all WRITING rows at completion
would publish another submission's unfinished values. A reusable heap array,
Swift `publications` capture or task record is not the required index-page lifetime.
No arbitrary index-storage limit may silently reduce which ready rows one scan
can encode; capacity and ownership must be established for the configured graph.

Successful physical completion publishes numerical output stamps and consumes
the configured input uses once. Literal error provenance is written separately
to configured metadata pages. A numerical function does not read link/global
error status, cancel downstream values or certify output by inspecting metadata.
The caller may reject a result or invalidate its table and submit a fresh complete
graph. Old callbacks, receive addresses and NIC-read completions must retain the
old table and physical storage; they cannot be redirected into the new table.

The current global CMP/ACK stream and mutable page-address lookup do not provide
that lifetime identity. Removing their epoch checks did not implement it. The
transport replacement must address the original table ownership directly, with
all routing/binding established before invocation; no recovery handshake or
numerical drain is allowed. Fleet reachability and ordinary verbs teardown remain
operational responsibilities outside numerical composition.

### Native interface evidence for the full storage binding

Read-only Objective-C runtime inspection of the installed M5 framework found:

| Class | Installed selector relevant to the required binding |
|---|---|
| `_ANEIOSurfaceObject` | `initWithIOSurface:startOffset:shouldRetain:`; `ioSurface`; `startOffset` |
| `_ANERequest` | `initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:`; `weightsBuffer`; `completionHandler` |
| `_ANEModel` | `mapper`; `intermediateBufferHandle`; `setIntermediateBufferHandle:` |
| `_ANEProgramIOSurfacesMapper` | `prepareANEMemoryMappingParams:request:`; `mapIOSurfacesWithModel:request:cacheInference:error:` |
| `_ANEBuffer` | `initWithIOSurfaceObject:symbolIndex:source:`; `ioSurfaceObject`; `symbolIndex` |

These observed selectors refine the interface investigation; they do not establish
ownership, accepted argument types, asynchronous behavior, or equivalence to the
winning MLE5Engine computation. In particular, a numeric intermediate-buffer
handle is not a CPU pointer or an IOSurface ID by inference. The current
`CoreMLParameter` passes input/output bindings only, and `CoreMLFunction.call`
retains its prediction wrapper, post-prediction identity assertion and completion
event. That path is not the required full-storage/native-completion binding.
Resolve the actual mapped storage and its lifetime in the connected replacement;
do not invent a private-buffer exception or replace ANE with a slower backend.
Bryngelson's mapped-surface account and the existing native references in
`algorithm-sources.md` motivate this binding; they do not prove these private
selectors' semantics on the installed systems.

### Contained acceptance run

After the connected source is written and excluded callers/definitions removed,
commit the complete source on main, synchronize both repositories on both machines,
and build every changed executable/library on both machines concurrently. Preserve
and identify the running bridge binary when transport changes require deployment.
Use the existing numerical/client runner, with its recorded real token array,
model/configuration identity, all 48 layers and one/two simultaneous invocations.
The run must include caller-owned invalidation and whole-NFE repetition evidence,
not transport recovery or a fault-dependent numerical branch. Compare against the
fastest validated local functions and collect the C34–C38 numerical, lifetime,
traffic and timing evidence in that same review. Review the entire resulting
source and trace against C01–C44 together; revise all uncovered issues before the
next acceptance run. No separate evaluator or old-runtime baseline result can
stand in for replacement execution.

## Source disposition for the connected change

| Source | Required disposition |
|---|---|
| `metal-microbench/reduce_scatter.swift` | Replace slot compiler, bound-call/prediction publication wrappers, old normalization maps, digest admission, status cancellation and settle/drain loops with the configured row caller. Reuse numerical functions, not its control structures. |
| `rdma/mesh-pages.c`, `rdma/mesh-pages.h` | Delete after all caller references migrate; no compatibility facade that preserves the excluded scheduler. |
| `rdma/mesh.h`, `rdma/mesh-client.c` | Audit every `mstream`, protocol frame, stream bitmap, retry/abort/close dependency; remove migrated computation mechanisms. Keep required page addressing and actual transport ownership only. |
| `rdma/mesh-functions.*`, `mesh-reduce.*`, `mesh-metal-executor.*`, `mesh-tensor.*`, `mesh-stream.c` | Trace remaining users before deletion. These are part of the dependency audit, not an unmeasured place to move the excluded implementation. |
| `rdma/mesh-dataflow.c`, `.h` | Finish the complete ABI and lifecycle above; unused replacements receive the same review as old code. Preserve the specified pairwise accumulator arithmetic; resolve input boundaries, publication ownership and table identity. |
| `rdma/mesh-flow.c`, `mesh-links.h` | Bind literal pages to supported verbs, preserve literal metadata, remove computation recovery and blocking issue paths. Review fleet accessibility separately from numerical execution. |
| `rdma/mesh-metal.m`, `.h` | Preserve aliases of actual registered storage; remove old page-runtime dependency after migration. Make row table, transmit and receive views usable by configured kernels. |
| `parameter_operations.swift`, `compute_backends.swift` | Explicit page views for all supported numerical operations; native callbacks publish values/metadata without a second control protocol. No dense fallback on the mesh path. |
| `matrix_operations.swift`, `matrix_shaders.swift`, attention/FFN numerical dependencies | Preserve winning arithmetic while using payload-safe views, FP32 contraction outputs and pre-realized geometry. Review scratch/constant ownership transitively. |
| `model_file.swift`, `parameter_configuration.swift`, model/artifact generators | Load final page storage and realize all ownership/layout/backend decisions before invocation. Count generated implementations; embedded native parameters remain an obligation. |
| `forward_bench.swift`, `demo_runtime.swift`, `tools/mesh/*`, build files | Route through the one caller; use caller-supplied inputs and existing metrics. Record commits, model/input/config identity, running bridge identity and both-machine builds. Remove obsolete runtime knobs after migration. |

## Literature and limits

Papadopoulos and Culler's [Monsoon (ISCA 1990)](https://www.cs.cmu.edu/~18742/papers/Papadopoulos1990.pdf)
provides operand matching and presence-state prior art. It does not prove that
this Swift loop has no scheduling overhead, or that the paper implemented this
repository's exact page table and batching policy. Dennis and Arvind/Nikhil are
the earlier dataflow/tagged-invocation references already in `distributed-reduce.md`.

Rabenseifner's [collective algorithms (ICCS 2004)](https://fs.hlrs.de/projects/rabenseifner/publ/myreduce_iccs2004_2.pdf)
and Patarasuk and Yuan's [bandwidth-optimal all-reduce (JPDC 2009)](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf)
support partitioned reduction and communication accounting. For the corresponding
all-reduce model, per-participant useful traffic is `2(n−1)S/n`; page headers,
topology, setup and launch overhead require separate accounting. These papers
do not make latency zero or guarantee a particular SoC speedup.

Saltzer, Reed and Clark's [end-to-end arguments (TOCS 1984)](https://web.mit.edu/saltzer/www/publications/endtoend/endtoend.pdf)
support locating final correctness checking at the endpoint. Caller-only NFE
repetition and non-short-circuiting asynchronous metadata are this repository's
explicit contract, not a claim that the paper specifies this ABI.

The existing NCCL, Active Messages, Legion/Realm, StarPU, PaRSEC, OmpSs, Naiad,
MapReduce/RDD and async tensor-parallel references describe related mechanisms;
none grants an exemption from the page-storage or caller restrictions here.
Their own control structures must not be imported merely because they are mature.

Apple's [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
documents Thunderbolt RDMA as two-sided SEND/RECV, without hardware-initiated
remote writes, using local-write memory registration. Thus one-sided RDMA_WRITE
is not an available implementation strategy on these machines. SEND/RECV into
literal registered pages is the applicable binding. Earlier repository prose
calling this actual substrate one-sided was incorrect.

Logical invalidation and physical revocation are different operations. A stale
device write cannot be stopped by clearing a software stamp; NVMe availability
can restore data but cannot revoke that write. The implementation must exhibit
the concrete ownership/revocation mechanism rather than infer it from any of the
papers above. Likewise zero latency, immediate reuse of hung-kernel addresses and
complete control of opaque backend allocations are not established by citations.

### Native backend contract found during this review

The installed macOS SDK's `CoreML.framework/Headers/MLPredictionOptions.h`,
lines 26–39, describes `outputBackings` as proposed output storage, and explicitly
allows framework allocation when a model does not support supplied backings.
`compute_backends.swift` loads embedded model parameters and sets only input and
output feature bindings. Its `backingUsed` assertion runs after prediction; it
does not establish parameter/intermediate ownership, and deleting it cannot make
an unsupported backing become the numerical output. The interface is documented
at [MLPredictionOptions.outputBackings](https://developer.apple.com/documentation/coreml/mlpredictionoptions/outputbackings).

C10 requires a demonstrated binding for the actual configured native operation,
including all its numerical storage. The current public wrapper does not provide
one. This is a concrete implementation gap, not evidence that all ANE interfaces
are impossible, nor authorization to fall back to a slower backend. Source review
must distinguish this unresolved interface from missing ordinary page arithmetic.

## Evidence ledger at the initial review

The metal repository's `docs/data/contraction_ranges_2026-09-09.json` records
30 dense local projection runs on both SoCs; all were finite, with partition
reconstruction relative RMS below 1e-6. Reconstruction was offline analysis,
not canonical mesh reduction. `parameter_fp32_outputs_2026-09-09.json` records
12 dense FFN output-precision cases, not page-backed parameter execution.
The earlier RDMA artifacts exercise the old caller. None closes replacement
requirements C01, C07, C18, C21, C25 or C28–C38.

`attention_weight_views_2026-09-09.json` records eight existing local parameter
evaluations at metal `22bbedf`, mesh `0261f16`: layers 0 and 5, 64 input rows,
heads 14–15, MPS and TensorOps on both SoCs. All outputs were finite. M5 backend
outputs matched exactly; M4 TensorOps-versus-MPS relative RMS was 0.000190541 and
0.000260375, respectively, with maximum absolute difference 0.001953125.
The inputs are normalized model embedding rows, with identical input hashes per
backend comparison. These runs use dense storage and do not establish C07 or C34.
The received/local-page normalization shader compiled into a compute pipeline on
both GPUs; it was not numerically executed in these runs.

The same artifact records a six-layer, 1024-row, two-in-flight RDMA regression
of the existing caller at metal `aadc851`, mesh `667aa41`. Both participants
completed six NFEs with 72 agreements, zero disagreements/nonfinite values/retries,
and settled exits. Their final logits hashes match. Measured invocation durations
were 149.834–150.613 ms on M5 and 148.196–148.395 ms on M4. The much smaller
7.732/11.657 ms completion gaps in `period_ms` are not NFE latency. This preserves
the existing numerical path through the weight-view change; it does not validate
the replacement transport, normalization binder, full 48-layer graph or latency
target. Full model-file SHA256 matched independently on both machines:
`5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d`.

Future evidence belongs in this matrix/ledger with exact source and input identity,
observed result and explicit limitations. Update current disposition rather than
append another narrative that leaves the active implementation ambiguous.
