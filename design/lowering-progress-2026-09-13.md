# Lowering implementation progress

The [nine-point plan](pallas-lowering-plan.md) remains active. This record covers
landed increments, not completion of the full plan or a general performance claim.

## Landed source increments

- `db5654b`: operation, derivative, dtype, shape and backend inventory in
  [lowering coverage](lowering-coverage.md).
- `b6583ff`: compiled integer indices, masks, indexed loads and selection; compiled
  embedding gathers; shared Xonotic two-dimensional comparisons and selects.
- `51145c5`: expression outputs retain separate actual input dependency lists.
  An absent operand for one output no longer delays a different independent output.
- `a7490b0`: Xonotic chooses shared lowering before emitting custom Metal code,
  removing eager emission's rejection of otherwise supported FP16 numerical paths.
- `dfb0d4b`: actual per-direction transport frame accounting and independently sized
  CQs. [Source proof and limits](actual-frame-capacity-2026-09-13.md).
- `6440f9c`, `13ed4bd`: existing streaming-overlap example migrated to native
  contractions, matching peer partitions and reusable slots; report serialization.
  [Migration contract](streaming-overlap-native-2026-09-13.md).
- `d1430f7`: [dynamic reader lifetime design](dynamic-reader-lifetimes.md), retaining
  selector/source retirement identity across independent source reuse. This design
  motivates ongoing implementation; a selected-readiness-only hook is insufficient.

## Operational evidence

[Raw observations and traces](../measurements/lowering-2026-09-13/) identify measured
source revisions. Both bridges were gracefully restarted with the new frame/CQ
implementation after confirming zero clients, preserving the 549,650,432-byte
registered region. The provider accepted separate CQs on both M5 Max and M4 Pro.

The gold chain passes on the two-node CPU and Metal paths with FP16 inputs. The
ordinary exact cancellation outputs remain 2 and 0. The new indexed expression
retains an int64 value of 2**53+1, masks its invalid table access, and produces
`[[16,18,20,22],[1,1,1,1]]`. A second output of the same expression kernel remains
absent until its separate input is supplied, then produces sixes. No Python
numerical callback is submitted in the recorded Metal gold.

The existing native overlap example also completed 15 warmup and 30 measured
invocations using three reusable slots, 4096-by-256 inputs and 16-row regions.
It configured 256 regions per invocation and completed 11520 native CPU numerical
submissions per participant across warmup and measurement. The three terminal
invocations checked against the numerical reference had maximum absolute error
5.508994e-6. This is terminal numerical evidence, not validation of every earlier
intermediate. Raw timing samples and online statistics are retained.

The overlap run exercises repeated storage reuse and a configured population
larger than the previous 255-message maximum-block cap. Latest-occurrence traces
do not establish peak historical occupancy, so it is not claimed to saturate or
measure the throughput benefit of the enlarged frame capacity. No matched old/new
polling-overhead conclusion follows from these runs.

## Remaining completion requirements

General indexed scatter reductions, general logical-rank lowering, legal fusion/storage planning,
collective placement and full caller migration remain unfinished. Xonotic source
migration is not a substitute for its operational numerical/performance evidence.
The plan's no-feature-regression and no-overhead acceptance audit remains open.
The dynamic cases below establish their specific selected-source and reuse behavior;
they do not establish arbitrary dynamic indexing or complete Pallas-style lowering.


## Dynamic selected sources and constant specialization

`5175c7e` and `d5c41c0` implement whole-Tensor indexed loads, numerical selectors,
and canonical selected-source lifetime retirement. `c9c38f8` extends the existing
gold example: selected source rows produce fours while an unrelated source is
absent; the selected source then publishes its next occurrence before the old
unselected source arrives, and the next output is sixes. Both CPU and Metal local
runs pass. This checks a reuse error that ordinary final FFN accuracy would miss.

`056ebce` retains explicit constant declarations and eliminates dynamic selectors
only when every candidate belongs to that declared constant storage. Present
produced tables still require selectors. `228ea96` moves trace export after the
dynamic example, so numerical submissions from that demonstration are included.
Latest-occurrence traces still do not reconstruct full historical retirement.

The archived `constant-cpu.json.gz` and `constant-metal.json.gz` observations use
`228ea96`, FP16 inputs, three measured gold invocations plus warmup and the existing
side checks. Both pass with maximum gold error 0.001953125; exact cancellation,
masked integer indexing and both dynamic generations also pass. Both report 1652
completed numerical submissions. A local CPU run before constant specialization
reported 1844; this records removed selector work, not a matched throughput claim.
CPU completion latency has count 3, mean 13.249736333333333 ms and sample variance
2.097980727796333 ms²; Metal count 3, mean 24.254124666666666 ms and sample variance
3.5864129249723318 ms². These are local runs, not new distributed scaling evidence.


## Direct CPU register accumulation and indexed diagnostics

`dd19ebe` replaces the CPU half contraction's scalar dot path with direct-stride
4×4 NEON register accumulation in FP32. No full operand conversion storage is
introduced; configuration retains loader functions and canonical output addresses.
The existing all-FP32 Accelerate path remains. See
[cpu-register-contraction](cpu-register-contraction.md) for mechanism and limits.

`b8ab668` and `bf6128d` expose original indexed reader, selector, retirement and
output map identities in the existing trace. `7bcf4f5` and `a442808` add produced
selector subranges with retained vector/range lifetimes. This range extension is
compiled, but its dynamic runtime evidence remains to be supplied by the scatter
vertical slice. `0184b6a` and `9462daf` give expression lowering its whole configured
grid once, enabling shared routing preparation without repeated per-output setup.

Archived `neon-strided.json.gz` uses installed library `bf6128d`. CPU passes the
FP16 gold, exact cancellation, masked indexing, dynamic reuse and transposed
3×5 by 5×7 tail contraction. CPU batch time is 6.537459 ms; completion count 3,
mean 5.939403333333334 ms, sample variance 0.2719680602963338 ms². Three subsequent
Metal runs pass the same numerical cases. This is numerical and operational
evidence, not a matched speedup or broad kernel performance characterization.

An initial Metal observation failed before the example checked the independent
tail result's readiness. The example previously assumed main-chain completion
implied unrelated numerical completion. `a442808` explicitly observes the side
results after the timed work. Source inspection found the physical MPS descriptors
and transpose flags correct; no backend fallback or extra numerical dependency
was introduced. The first failure did not record the result values, so it alone
cannot distinguish unpublished output from numerical mismatch; the corrected
observer and three subsequent passing runs are the retained evidence.


The subsequent two-node CPU gold also passes with maximum error 0.001953125,
including cancellation, strided tails and dynamic indexed reuse. Raw observations
and both participants' traces are in `neon-paired-cpu.json.gz`, with each installed
library revision retained explicitly (`bf6128d` local, `4be627c` peer). The local
completion count is 3, mean 6.897945 ms, sample variance 0.000019781713 ms², and
batch time 6.90275 ms. These unmatched observations do not establish a distributed
speedup over the improved local baseline. The peer exited normally via SIGTERM
after the local numerical validation completed.


## Executable segmented indexed addition

`ae4183f` implements vector-row indexed addition through the existing expression
and kernel_call API. It groups retained destination keys/original ordinals,
computes per-segment partials in FP32 and selects only the partials contributing
to each final destination region. `25cfbd7` adds defined unsigned modular integer
accumulation. See [scatter lowering](scatter-lowering.md) for mechanism and limits.

The existing gold example now supplies three routing chunks, withholds the last
update chunk and observes the pointwise consumers of destinations 0, 1 and 3.
Destination 2 remains unavailable until its missing payload is supplied. Duplicate
keys across chunks sum correctly, the masked update is excluded, and the empty
destination produces zero. The second occurrence changes both routing and values
using the same storage, giving consumer rows [10, 0, 18, 12] after [8, 0, 14, 8].
All four rows repeat their scalar value across four features. CPU and Metal both
pass. This exercises produced selector ranges and their shared-vector lifetimes,
not just arithmetic over constant indices. Host observations occur after the
main timed gold and introduce no numerical dependency between independent rows.

Raw `scatter-reuse-cpu.json.gz` and `scatter-reuse-metal.json.gz` retain installed
library and example revisions, numerical observations and indexed trace maps.
Their observer times are upper bounds on availability and include host writes;
they are not physical compute/wire occupancy or matched performance gains.
Coverage is six updates, four features and four destinations over two occurrences.
General rank, wider radix tiles, production-scale memory/launch costs, fused
update expressions and the actual Xonotic migration remain open. Partial storage
is bounded by maximum segment count, but separately allocated canonical quanta
and per-reader candidate metadata still require storage/lowering optimization.


## Xonotic consumer migration and destination validity

`9b4ea55` migrates rank-one Xonotic scatter, scalar/vector pointwise expressions
and sum consumers. Remaining custom kernels retain explicit block pointers and
strides rather than demanding a single whole-region view. No dense operand copy
is introduced. [Xonotic source mapping](xonotic-indexed-add.md) records remaining
nonstreamed operations and axis coverage. This is an incremental migration, not
completion of the expert/neighborhood and derivative inventory.

The existing two-node planner completed normally at source `9b4ea55`: width 16,
eight bots, eight-row tiles, five-second duration, 3384 returned plans and 559
objective switches. `xonotic-planner.json` retains both outputs and configuration.
This validates the actual planner's compilation, transport and repeated execution;
it does not exercise every newly migrated scatter consumer or establish numerical
residuals, a throughput gain, or full Xonotic coverage. The peer Xonotic environment
uses Python 3.12 and the repository's declared MLX/NumPy dependency versions.

`e4d18c0` evaluates scatter destination validity before narrowing indices to U32.
`9a96306` extends the second existing scatter occurrence with index 2**32+2;
it must not wrap into destination 2. Both CPU and Metal pass, producing second
consumer rows [10, 0, 12, 12], rather than the earlier case's [10, 0, 18, 12].
`scatter-valid-cpu.json.gz` and `scatter-valid-metal.json.gz` retain these runs.
The earlier reuse artifacts remain evidence for their separately recorded inputs.


## Fused pointwise updates

`c4086e5` lowers pointwise update expressions directly inside segmented indexed
accumulation, using the same scalar emitter as ordinary expressions. Every used
operand retains its own selected source region and lifetime. Broadcast values
and mixed half/float sources preserve FP32 arithmetic; declared constants avoid
unnecessary dynamic lifetime bindings. There is no transformed-update tensor or
extra pointwise launch. Nested indexed loads and feature reductions inside the
update expression remain unfinished forms, not an eager-load fallback.

The existing example now uses updates * factors + 1 with separately streamed
FP32 factor rows. It supplies the final update rows before their factors; the
final consumer remains unavailable until the factor rows arrive. Independent
consumers have already completed. Both CPU and Metal pass two occurrences with
changing routes and factors: consumer rows [20, 0, 32, 18], then [32, 0, 38, 40].
Each row repeats its value across four features. Gold, cancellation, strided and
indexed reuse checks also pass. Archived scatter-fusion CPU/Metal observations
and traces retain exact library/example revisions. Both report 1704 completed
numerical submissions, the same count as the prior unfused expression case;
this establishes absence of an added transform launch in the covered program,
not a matched throughput improvement.


## Shared contraction ownership and reader fanout

`89fec3f` moves K partitioning, FP32 partial contractions, pairwise reduction and
final casts into the shared `kernels.dot` expression lowering. `nn.linear` now
supplies output layout and placement. Existing CPU/MPS/Core ML operation owners
remain unchanged. `55eddbd` fixes direct mapped BlockSpecs by resolving their index
maps per coordinate, including mixed mapped/whole operands. The existing example
adds reversed output rows from transposed input views with a ragged K tail. Both
CPU and Metal pass that numerical check, the original cancellation case and gold.
This migration preserves the numerical launch formula for each output tile;
matched hardware nonregression remains a separate acceptance requirement.

`d0a12e2` documents [shared routing ownership](shared-routing-ownership.md).
`d8386f5` and `4ca4369` implement canonical reader groups for fanout that cannot fit
the direct reader-plane representation, with one actual member identity per
source row and reader occurrence. Direct planes remain the setup-selected path
when they suffice. `e5a7cd6` exposes those retained identities in Python traces.
`449ab9e` releases member storage with its owning program/maps and makes the
existing metadata owner the sole member-reset owner for both host and remote
publications. No new numerical launch, source copy, epoch guess or invocation
rendezvous implements group completion. Program teardown remains outside numerical
invocation. This source ownership fix is not a claim of a measured graph-churn
memory curve.

The configurable existing scatter case passes on both backends with 67 updates
in 33-row tiles, including a ragged final tile, SIMD radix work beyond one group,
FP32 fused factors, invalid/masked indices and two changing occurrences. Earlier
traces before the added fanout case show zero grouped readers for the six-update
program and two grouped source rows for the wider program, each with 166 distinct
retained member identities. Source grouping removes the reader-plane capacity
failure; it does not remove worst-case destination-by-candidate metadata or the
per-potential-segment physical allocation and launch costs.

The same example also sends a source through RDMA to 65 independent numerical
consumers and returns their results. Both CPU and Metal complete two source
occurrences: first/last branch outputs 2/130, then 3/195, with every branch checked.
The peer trace joins the grouped source row directly to a receive transfer with
two occurrences and 65 distinct member identities. This exercises remote receive
reset, not merely local directory reuse. Both peer workloads terminated normally.

Raw `reader-lifecycle-local-*` and `reader-lifecycle-paired-*` archives retain each
installed library revision, example revision, observations and both participants'
traces where applicable. The paired CPU gold maximum error is 0.001953125; Metal
is 0.00390625 under the unchanged absolute-plus-relative tolerance. These runs
establish covered execution and reuse, not matched throughput improvements or
physical GPU/wire overlap percentages. Nested dot epilogues, arbitrary-rank
lowering, sparse routing ownership, production storage/launch optimization, full
caller/derivative migration and the final performance audit remain unfinished.

## Shared sparse domain follow-up

The [sparse routing source and operational record](sparse-routing-progress-2026-09-13.md)
documents the shared candidate/directory ownership implementation through
`5843a8b`. It replaces per-destination candidate descriptors and duplicated
directory/table readers with explicit shared domain holds. Local CPU/Metal pass
1025 updates and 17 destinations; paired CPU/Metal pass the existing float16
gold with remote fanout reuse. The actual Xonotic gather/concatenate graph now
demonstrates independent indexed producers and consumers through the shared
representation. Source, precision, locality, derivative and performance limits
are recorded with the raw evidence. The bounded active-segment increment follows
below; physical partial storage and the broader plan remain unfinished.


## Bounded active numerical grids

`afd3b7b` retains the static maximum min(chunk updates, destination rows) for
scatter partial capacity. `b081848` implements native count/slot activation and
explicit produced/omitted dispositions; `4975a24` binds exact producer identities
in shared lowering; `dc4c80a` exposes count, omission and disposition reader traces.
Omission performs no numerical launch and publishes no payload. Routes consume
its explicit disposition without reading a fictitious payload occurrence.

The implemented indexed contract requires count and all selector/range views to
belong to one actual produced directory output, with compiler-supplied empty
inactive ranges. Existing indexed lifetimes retain late unused payloads;
ordinary unused reads retain separate retirement facts. The count and disposition
cannot recycle before their actual readers finish. Independent indexed-count
producers and arbitrary inactive selections remain outside this increment.

[Active-grid source and evidence](active-segments.md#archived-operational-observations)
now records local CPU/Metal passes for 1025 updates in 513-row chunks and 4097
updates in 2049-row chunks, both with 17 destinations. Installed library is
`dc4c80a`; examples are respectively `dc4c80a` and `e0ecb91`. Raw unchanged logs
and traces are archived with [explicit provenance](../measurements/lowering-2026-09-13/active-omission-provenance.json).
Four normal/changed/all-invalid/normal occurrences check independent consumers,
all-empty output before any update/factor payload, subsequent late-unused
retirement, and reuse. The larger case records the count, bounds and selector
on distinct directory pages under the same produced output ownership.

Each trace's 34 main scatter partial functions account for 50 numerical launches
and 86 omissions over 136 slot occurrences. Six Xonotic gradient partial functions
add 12 launches; whole-workflow submissions/completions are 2067/2067. The same
records include independent gradient outputs before an unrelated cotangent
arrives, without allocating the primal operand. This is bounded numerical and
source-lifetime evidence, not a matched throughput gain or complete derivative
coverage. Finer physical partial storage, repeated readiness traversal, general
rank/fusion, placement and the plan's final no-regression audit remain open.

## Composed contraction epilogues

`3288be4` replaces root-only dot dispatch with recursive pointwise epilogue
lowering in the existing expression compiler. Native FP32 K-panel contractions
and intermediate pairwise reductions are shared by identical contractions in
one grid coordinate. Repeated occurrences also retain the same expression input
identities. The last one or two partials feed the existing scalar emitter,
combining final summation, pointwise work and output conversion in one function.
A separately requested bare contraction retains independent publication and
canonical reader lifetime; it does not acquire the epilogue's extra operands.
Computed contraction operands and shape-changing dot reductions remain open.

The existing streaming-algebra example at `e8ebac1` requests
`expression(dot(a,b)*2+bias, dot(a,b))` over three independently supplied rows,
with a ragged two-panel K contraction and transposed weight backing. It supplies
row 1 without bias, observes the bare contraction while its epilogue remains
unavailable, then supplies that bias and observes the epilogue while rows 0 and
2 remain unpublished. It finishes those rows and repeats using the same storage.
Both CPU and Metal match exact dyadic reference values in both occurrences.

The native trace retains two shared contraction-panel functions, one fused
epilogue and one bare-result addition per row: twelve configured functions and
24 numerical submissions over two occurrences. For row 1 in both local traces,
partials start at canonical rows 196 and 200; epilogue output 168 reads those
and bias 152, while bare output 180 reads only the two partials. This is direct
dependency evidence, not a reconstruction from timing. No scalar callback,
extra host operand store or invocation-time allocation implements this change.
The remaining K-panel storage and native launch costs have not disappeared.

Local CPU/Metal runs use 4097 scatter updates, 2049-row chunks, 17 destinations,
three gold invocations and the existing Xonotic forward/derivative cases.
Both finish 2091 numerical submissions; unchanged gold maximum absolute errors
are approximately 1.65e-6 and 1.85e-6 respectively. `composed-provenance.json`
and compressed `composed-{cpu,metal}` observations/traces retain source and
configuration under `measurements/lowering-2026-09-13`. These observations do
not establish matched throughput improvement or complete general composition.

The paired CPU and Metal float16 runs also pass at example `88cba6a` with
67 updates in 33-row chunks and 17 destinations. Installed local library is
`e8ebac1`; installed peer library is `88cba6a` (identical numerical source).
The distributed gold maximum errors are 0.001953125 and 0.00390625 under the
unchanged tolerance, and all 65 remote fanout branches match both occurrences.
Rank zero finishes 1357 submissions in each backend. Indexed and composed-dot
side cases execute on rank zero; these runs do not establish remote scatter or
remote composed-dot candidate lifetime coverage. Both peer processes terminate
normally with SIGTERM after writing their traces. The paired observations,
both participants' traces and exact provenance are archived alongside the local
records. Broad expression/rank lowering, remaining caller migrations, storage
and launch optimization, and matched end-to-end performance remain open.

## Demand-driven computed contraction operands

`2aec0ae` replaces root-reference-only contraction construction with one
`_ContractionRegions` setup owner. Natural shapes, mapped/local versus whole
coordinate domains and backing cuts remain distinct metadata. An outer K-panel
requests only its left M×K and right K×N regions. Pointwise computations and
nested contractions recursively produce those canonical regions, preserving
FP32 intermediate values and independently ready native matmul inputs. Shared
region demands retain expression, source-view and origin identities; specialized
program IDs participate in those identities. No runtime graph builder or hidden
whole-intermediate operand is introduced.

The existing example now expresses a linear/swish/linear chain in one numerical
expression. X is 3×4 in single-row blocks, W1 is 4×6 in two-column blocks, and
W2 is 6×3 in two-row blocks. Both dots have K tile two. Supplying X row one,
W1's first feature panel and W2's first contraction panel permits the second
matmul to finish a contribution before the other two hidden panels exist.
The observer joins W2's actual canonical row range to its configured numerical
consumers and records their completion; it does not guess which function ran
from command order. After the missing weight panels arrive, row one finishes
while X rows zero and two remain unpublished. Both complete occurrences match
the independent float64 reference within the unchanged FP32 tolerance.

CPU and Metal local runs pass with installed source `7fc905d` and example
`ff3afa3`, before the subsequent FFN caller and mixed-dtype changes. Native
function 42 consumes W2's first canonical block (first row 236, four allocated
pages) in both early observations. Whole-workflow submissions are 2175, all
completed. The example initially shadowed its final timing dictionary with the
new consumer observation tuple; `ff3afa3` fixes that reporting error without
changing numerical execution. `nested-initial-provenance.json` and the compressed
`nested-final-{cpu,metal}` records preserve the passing runs. These establish
partial producer and partial consumer execution for this composition, not
matched throughput parity, remote nested contractions, or arbitrary-rank lowering.

Computed indexed loads, shape-changing reductions and mismatched non-singleton
mapped/global operand domains still require broader lowering. Native M/N panels
still need to fit their actual backing blocks. These limitations remain explicit;
there is no whole-operand copy or wait fallback for them.

## FFN caller composition and mixed operand types

`fab4ada` moves FFN hidden projections, their balanced partition sum and swish
into one shared expression call. It preserves the prior row/column partition
cuts, the activation's declared input dtype (including FP16 rounding), the
existing exchange boundary, and FP32 down-projection accumulation. The caller
no longer constructs separate completed hidden contractions and hidden sums.
An explicitly nested expression instead retains computed panels in FP32;
adding an expression-level cast remains necessary to spell the former FP16
rounding boundary inside an entirely nested FFN expression.

`1448f0e` removes the native contraction operand-dtype equality restriction.
The existing CPU implementation already has independent typed loaders and FP32
accumulators; MPS retains each original matrix buffer, offset, strides and dtype.
No conversion buffer or alternative numerical backend implements the change.
Core ML rectangle metadata and its generator now retain each operand's dtype
separately instead of reconstructing the right type from the left. Its existing
FP32 arithmetic conversion remains unchanged. Native libraries and the generator
compile; Core ML runtime/placement/internal-copy claims are not established by
that source evidence.

Installed source and example `1448f0e` pass the integrated CPU and Metal local
runs in both float32 and float16, plus the float16 two-node gold and remote
fanout workflow. The float16 nested expression specifically exercises
FP16×FP16→FP32, FP32 swish, then FP32×FP16→FP32 through native MPS on the Metal
backend. It observes the second contraction completing a partial before two
hidden panels arrive in each occurrence. This is evidence for the covered
heterogeneous MPS input combination, not a universal undocumented type guarantee.

The unchanged 4097-update float32 configuration completes 1919 submissions on
each backend, compared with 2175 before the FFN migration: 256 removed numerical
submissions. Float32 gold maximum errors are approximately 1.65e-6 on CPU and
1.58e-6 on Metal. Local float16 runs complete 2035 submissions, and paired rank
zero completes 1249; all four float16 runs have gold maximum error 0.001953125
under the unchanged tolerance. The paired gold still streams across the real
link and its 65 remote fanout branches reuse their inputs correctly. Nested
and indexed side operations remain local to rank zero. Both peer workloads
terminate normally after recording their traces.

`nested-integrated-provenance.json` and compressed `nested-integrated`,
`nested-mixed` and `nested-paired` observations/traces preserve configurations,
source and both participants where applicable. No matched throughput improvement
is inferred from the submission reduction or these short timings. General rank,
indexed/reduction composition, remaining derivatives and callers, storage/launch
optimization, collective placement and full performance acceptance remain open.

## Shared reductions and caller removal

`70b6511` extends the existing region owner to row reductions and renames it
`_ExpressionRegions`. A reduction demands the full resolved child feature domain,
computes each available feature partial independently, combines actual siblings
in a balanced tree, and caches the resulting row statistic across output tiles.
A computed dot can consume that cached statistic directly. Square-and-sum is
fused in each feature partial; there is no squared-tensor intermediate or
whole-tensor prerequisite. Scalar/intrinsic-only expressions have scalar domains;
indexed reduction domains must be supplied by explicit index-shaped operands.

`82c1ab3` expresses RMSNorm as one shared expression with the original output
cuts and dtype. `4841436` migrates the remaining Xonotic vector and matrix-row
sum/mean callers and deletes `_row_reduce` and its unused row-map helper.
Real means divide the FP32 sum before final conversion, eliminating intermediate
half rounding/overflow. Integer means retain their typed sum boundary before
integer division. These precision choices are documented in the canonical source
record, rather than hidden in caller-specific partial constructors.

`1aeb6cb` uses unsigned modular addition for integer-valued signed reduction
accumulators and partial merges, restoring signed interpretation at use. The
first Metal run then exposed an unsupported `simd_sum(ulong)` overload at
compilation. `b844daf` uses three supported UInt32 SIMD sums with exact limb
carry reconstruction; its modular arithmetic proof is in algorithm-sources.md.
The floating path is unchanged. The compile-failure log is retained alongside
the successful runs; no numerical fallback implements the correction.

The existing workflow supplies two of three feature panels for one normalization
row. A reduction consumer completes before the last panel arrives; afterward
that row normalizes while two unrelated rows remain unpublished. Both repeated
occurrences pass. The Xonotic graph now also requests matrix row means followed
by a vector sum, observing the early row means before the unrelated source block
arrives; the totals are 30 and 158. Integer cases preserve cancellation beyond
2**53 and signed 64-bit wrap. The final paired case also exercises low/middle
limb carries across SIMD lanes and overflow between separate feature partials,
returning 65536, 4294967298, -9223372036854775808 and 9223372036854775807.

Local CPU and Metal float32 workflows pass, including the 4097-update scatter
case. Both report 2045 completed submissions. Paired CPU and Metal float16 runs
report 1343 completed submissions on rank zero and gold maximum error
0.001953125, with remote fanout reuse intact. The paired reduction observer joins
canonical partial rows 516–523 to consumer function 79 in both occurrences.
Nested/reduction/indexed side cases remain local to rank zero; paired gold and
fanout exercise the actual link. Both peer workloads terminate normally.

`reduction-provenance.json` and compressed `reduction-*` observations/traces in
`measurements/lowering-2026-09-13` retain exact library/example revisions and
both participants. Before-migration observations use a smaller example scope;
added integer and Xonotic cases account for additional work, so total counts and
short timings do not establish a throughput comparison. General logical rank,
remaining indexed/derivative composition, physical storage/launch optimization,
collective placement and matched performance acceptance remain unfinished.

## Explicit casts and nested local FFNs

`7d6b8b4` adds explicit expression conversion boundaries and typed canonical
contraction panels. `0c441b5` composes the local FFN through its hidden activation
into its down projection, preserving the deliberate hidden FP16 conversion.
The exchange configuration still publishes the hidden panels and substitutes the
returned references into the same down-projection formula. Xonotic cast nodes
use the shared expression conversion. No second numerical executor is introduced.

`1bd8c82` updates the existing operational example to exercise the actual local
configuration with `exchange=None`. It adds a rounded and unrounded nested
projection, scalar half rounding, and the difference between converting before
and after summation. CPU float32 completed 1942 submissions with gold maximum
absolute error 1.654e-6; the scalar results are `[0,0,1]` and the two sums are
`1,0`. These observations are archived as `typed-initial-cpu*` with both library
and example at `1bd8c82`.

The corresponding Metal run exposed an MPS restriction, retained verbatim in
`typed-metal-mixed-failure.log.gz`: mixed multiplication requires FP32 left and
output matrices and an FP16 right matrix. The earlier opposite-order coverage
was insufficient. The explicit rounded hidden operand followed by FP32 weights
exercises the unsupported order. This is a native layout-lowering issue, not a
reason to remove the expression's typed conversion.

`4c87c68` strengthens early-progress observation by tracing each nested output's
actual row ancestors separately and requiring a completed projection consumer
for each branch while the other hidden panels remain absent. `2c7928f` adds a
rectangular 3-by-5 times 5-by-7 mixed contraction in both operand orders with a
ragged K panel. These additions belong to the existing operational example;
they distinguish incorrect layout lowering from vector-only numerical agreement.

`192ff70` fixes reverse mixed contraction layout using
`AB = (BᵀAᵀ)ᵀ`. Setup allocates its FP32 partial as a transposed view of canonical
storage; native binding normalizes operand and output views before deriving
geometry and dependencies. MPS receives its supported type order. Source retains
the original operand pointers, byte offsets and strides: there is no operand
conversion copy, scalar fallback, or per-row multiplication workaround. The
existing sum/epilogue consumes the logical view. The raw native row-major
reverse-mixed binding remains constrained by MPS; the shared expression owner
provides the required layout. Core ML internal-copy and device-placement claims
remain outside this evidence.

The next logical-indexing migration is concrete: `cast_header.scale_route` uses
`take_along_axis`, and `strategy.row_gram_context` produces rank-three gathers
and slices. Shared lowering needs logical shape separate from physical matrix
shape and exact integer quotient/remainder for coordinate mapping. Nonnegative
flat ordinals decode using positive dimensions; negative user indices normalize
in signed arithmetic before address conversion. General gather derivatives also
need scalar-address collision reduction and must depend on cotangents and indices,
not unused primal data. A forward-only migration will not discharge that work.

Final library/example source `192ff70` passes both CPU and Metal locally in
float32 (1974 completed submissions) and float16 (2026), and paired over RDMA in
float16 (1336 rank-zero completions). Local float32 gold maximum absolute error
is 1.654e-6 CPU and 1.576e-6 Metal; all float16 gold runs report 0.001953125.
Both mixed rectangular results exactly match their representable numerical
reference. Both explicit cast-order results match, and both nested projection
branches complete useful work with hidden panels 1 and 2 withheld in both
successive generations. In the final Metal float32 trace, functions 50 and 62
read the first projection rows and separate unrounded/rounded hidden rows;
neither includes the withheld panels. Their native completions are observed
separately through the corresponding output's actual ancestry.

`typed-provenance.json` records each configuration and its compressed logs and
traces. Paired CPU and Metal peer applications exit cleanly after SIGTERM. Only
gold and fanout traverse RDMA; nested/indexed/reduction side cases remain local
to rank zero. These short observations and differing example scopes establish
neither matched throughput parity nor physical wire/GPU utilization. General
logical rank, remaining caller/derivative migrations, direct matmul binding
unification, launch/storage optimization, collective placement and matched
performance acceptance remain open.

## Logical rank and index-map caller migration

`560b74e` adds `argument.reshape(logical_shape).at(*coordinates)` to the existing
expression owner. The logical shape is setup metadata. Its coordinates flatten
to an ordinal and map onto the actual bound physical Tensor/Ref shape with exact
integer quotient and remainder. Physical block pointers, offsets and strides
remain the same. Per-axis validity is applied before flattening, so an invalid
logical coordinate cannot alias another valid physical position. Single inferred
axes and scalar logical shapes use the same setup resolution. This does not add
arbitrary-rank contraction/reduction semantics or general indexed scatter values.

`fac669c` routes forward gather, take_along_axis, transpose and concatenation
through this map for arbitrary logical rank, deleting their custom forward
emitters and previous rank/shaped restrictions. These operations no longer call
`matrix_view` to assemble or reinterpret a whole operand. Requested numerical
outputs still have canonical storage; this is not a claim that every transpose
or concatenation has been reduced to a metadata alias. Graph reshape continues
to retain its input storage. Existing derivatives remain, including the shared
row-gather derivative; general gather/take derivatives still need migration.
The take derivative's source-singleton addressing fix has source evidence only.

`ff54af8` fixes take_along_axis's symbolic non-axis broadcast shape and validates
its rank, integer indices and axis during setup. It extends the existing optional
Xonotic graph with rank-three takes, a broadcast take, inserted-axis gather,
transpose, concatenation, and a reshape/gather whose logical last dimension differs
from the backing tensor's physical width. `2d75409` corrects the integer arithmetic
observation's BlockSpecs to bind mapped scalar operands. `57e38fe` includes the
new logical outputs' actual page identities in the existing binding diagnostics.

CPU and Metal float32 local runs pass the complete example, reporting 2160 and
2162 completed submissions respectively. CPU and Metal paired float16 runs also
pass, reporting 1524 and 1525 rank-zero completions. Gold maximum absolute errors
are 1.654e-6 CPU and 1.576e-6 Metal locally, and 0.001953125 on both paired runs.
For both generations every logical operation publishes its first half with the
other source/index regions absent, then completes its remaining outputs after
those operands arrive. The broadcast take retains the same early source region
for later index consumers without delaying its early output. Signed and unsigned
quotient/remainder comparisons include values above 2**53, INT64_MIN, UINT64_MAX,
and negative divisors. Divisor zero and unrepresentable signed quotient cases
remain outside the documented arithmetic domain. The logical bound observation
returns `[0,1,-1,-1]`, demonstrating that invalid inner-axis coordinates do not
read adjacent physical values.

`logical-provenance.json` records exact installed library and example revisions,
configuration and raw compressed logs/traces. The final paired Metal binding log
names each new output's actual rows, and native trace descriptors retain its
index inputs and candidate source identities. Paired gold and fanout exercise
actual RDMA; all logical-indexing side operations execute on rank zero and use
float32 data. No distributed logical-indexing or matched throughput claim follows.
Both peer applications terminate normally after SIGTERM.

Remaining work includes static index-map realization without unnecessary selector
launches, broader axis/contraction/reduction composition, general indexed scatter
values and derivatives, remaining caller migration, storage/launch optimization,
collective placement and matched performance acceptance. No end-to-end performance
parity is inferred from these finite numerical and progress observations.

## Setup specialization of vector addresses

`0d6115d` realizes setup-known indexed addresses as ordinary Ref bindings in the
shared expression owner. `604082e` preserves source order when assembling access
masks, removing hash-order-dependent specialization. The numerical graph stays
fixed: gathers operate on vectors of indices; masks express numerical validity.
`c1103f8` keeps the operational example straight-line, with static and indexed
reads of the same tensor composed in one expression. Its distinct row values
expose incorrect alias selection across repeated publication.

Source review establishes that specialization reads setup expressions, not
operand contents, resolves selected entries in the existing Tensor block table,
and binds their existing offsets, strides and storage. Purely static accesses
need neither selector-producing functions nor selector storage. Index-vector
accesses retain the shared indexed representation. Evaluation uses bounded
4096-point chunks and documented fixed-width integer semantics; expressions
outside its proven domain retain indexed lowering. Reduction access analysis
uses the reduced domain rather than just the output width. No new numerical
scheduler or alternate operand store is introduced. The mechanism and prior art
are documented in [Static indexed access specialization](algorithm-sources.md#static-indexed-access-specialization).

The final `604082e` local CPU and Metal float32 runs each configure 1503 functions,
55 with indexed descriptors, and complete 2095 submissions. The same example
with the pre-specialization library configures 1545 functions, 89 indexed, and
completes 2216 submissions on each backend. Native trace inspection confirms that
the first transpose, concatenate and reshape/gather outputs now use ordinary
source-page inputs with zero indexed descriptors. Take retains indexed accesses
because it consumes index vectors. Initial `f1feeaa` measurements had different
CPU/Metal specialization counts; they are retained with provenance and superseded
by the mask-order fix and final runs.

Both final paired float16 runs configure 1439 rank-zero functions, 55 indexed,
and complete 1410 submissions. Numerical comparisons, early outputs with unrelated
source/index regions absent, static/indexed alias reuse, and the existing gold
chain and fanout pass on CPU and Metal. Local gold maximum absolute errors are
1.654e-6 CPU and 1.576e-6 Metal; paired gold error is 0.001953125 on both. Peer
applications terminate normally after SIGTERM. Gold and fanout traverse actual
RDMA; the indexed side operations execute on rank zero.

`measurements/lowering-2026-09-13/static-provenance.json` records installed library
and example revisions, counts, and compressed raw logs/traces. Each timing
summary includes count, mean and sample variance. These three-invocation samples
establish no end-to-end speedup: final local CPU mean completion is 3.968 ms
versus 4.395 ms before, while Metal is 18.896 ms versus 16.606 ms before. Fewer
configured functions do not alone establish recovered performance. Broader
contraction/reduction axes, general indexed scatter values and derivatives,
remaining caller migration, storage/launch optimization, collective placement,
and matched performance acceptance remain open.

## Xonotic integer matrix row reductions

`4f0beba` removes the real-dtype-only condition on Xonotic's matrix last-axis
sum/mean dispatch. These operations now share the existing vector and real-row
reduction owner. Source inspection establishes that the original matrix blocks
are retained and each row requests its own feature partials; integer matrices
no longer fall through to the custom whole-operand reduction emitter. Existing
integer-vector mean division behavior is retained. General axis reductions and
other reduction families still require migration.

The existing integer cancellation example now supplies one row at a time through
canonical writes and observes its direct sum and Xonotic sum before supplying
the next row. Local CPU and Metal runs both return exactly 65536, 4294967298,
INT64_MIN and INT64_MAX through both callers, including values above 2**53 and
modular overflow. Both local runs complete 2087 submissions. The paired Metal
float16 gold workflow also passes; its integer side observation still uses int64
on rank zero. The paired peer terminates normally after SIGTERM.

`integer-rows-provenance.json` retains raw logs/traces, exact installed library
and caller revisions, and timing count/mean/sample variance. The source and these
observations establish the migrated sum's numerical and independent-row behavior.
Integer means, boolean matrices and unsigned matrix rows have source-path coverage
only in this increment. No matched pre-migration matrix timing baseline was
collected, and no latency or throughput improvement is claimed. The remaining
nine-step plan stays open.

## Matrix column reductions through existing page views

`47ed3e8` routes Xonotic matrix axis-zero sum/mean through the same shared
reduction as axis one. Setup transposes the operand view, binds the existing row
reduction, and transposes the output view back. Inspection of `Tensor.T`, `Ref.T`
and `mesh_view_transpose` establishes that handles, extents and offsets are
retained, with dimensions and strides exchanged. No transpose kernel or copied
operand is introduced. The shared region owner partitions the original reduction
axis at backing boundaries and retains independent output-column dependencies.

The existing integer observation binds its actual pages both as a matrix and as
a transposed matrix. Direct row sums, Xonotic row sums and axis-zero sums of the
transposed view all return exactly 65536, 4294967298, INT64_MIN and INT64_MAX, one
output at a time while later inputs remain absent. `a8123ed` also adds axis-zero
means over the ordinary row-major source. Those means correctly remain incomplete
while a contributing source row block is absent; unrelated completed row means
remain usable. Two generations return column means [6,7,8,9] and [38,39,40,41].

Local CPU and Metal float32 runs pass, as does the paired Metal float16 gold
workflow. Integer side cases use int64 and column means use float32 on rank zero
in every run; gold and fanout traverse actual RDMA. The peer exits normally after
SIGTERM. `column-reduction-provenance.json` records the installed library and
caller revisions, raw compressed logs/traces and timing summaries. No matched
pre-migration column-reduction timing baseline was collected; no speedup follows
from this evidence. Multi-axis and broader logical-rank reductions, other
reduction families, general scatter composition, remaining callers and the
nine-step plan's performance acceptance remain open.

## Composable transpose and complete matrix sum axes

`fd74955` adds `.T` and `.sum(axis=...)` to the shared expression owner, with the
existing default of axis one. Axis zero and both-axis reductions compose through
transposition and the existing tiled sums. Xonotic now routes all nonempty vector
and matrix sum/mean axis sets through this owner, including full matrix sums and
means. Logical rank above two and other reduction families remain separate work.
A transposed output root writes through the destination's transposed view, avoiding
an extra output-copy function. `ea0b245` preserves the previous independently
published reduction output sizes after this caller migration.

Initial numerical evaluation exposed a computed-panel dtype defect: the old
unconditional FP32 temporary lost exact integer values before reduction.
`96033b5` makes computed panel storage follow expression dtype. `f361533`
distributes transpose through pointwise expressions and reverses contraction
operands at setup, preserving fusion over existing transposed source views.
Double transpose cancels. Intermediate full-sum statistics preserve accumulator
precision across both axes. This is a setup representation change, with the
same native page-stamp invocation and registered operand storage.

The final local CPU and Metal float32 examples each complete 2142 submissions.
Direct and Xonotic row/column integer sums preserve exact cancellation and modular
overflow; the full matrix sum returns 4295032833. The floating full-matrix mean
composition returns 30 and 158 after its scalar multiplier in two generations.
Existing independently ready row outputs, column means, delayed indexed inputs,
scatter, mixed contractions and fanout continue to pass. The paired Metal
float16 gold workflow also passes. The new integer and mean side cases execute
on rank zero, while gold and fanout cross actual RDMA. The peer exits normally
after SIGTERM.

`axes-provenance.json` records final installed library and caller revisions,
configuration, compressed logs/traces, and timing count/mean/sample variance.
These examples do not establish a matched latency/throughput improvement or
complete coverage of arbitrary transposed contractions. Load-valued scatter
integration was reviewed with the scatter agent; the bounded-loop and dependency
requirements are recorded in `scatter-lowering.md`. That integration, broader
rank/axis operations and derivatives, remaining caller migration, storage/launch
optimization, collective placement and matched performance acceptance remain open.

## Logical-rank pointwise caller migration

`e61da95` routes supported Xonotic pointwise arithmetic, comparisons, casts,
selection and unary expressions through shared lowering across logical ranks.
Already-aligned matrix views retain their existing direct Ref bindings; other
logical layouts use right-aligned broadcast coordinates over the original tensor
pages. These operations no longer reach the custom whole-operand Metal emitter.
`b8bdc6a` shares output coordinate construction with indexed callers. `d895f62`
removes the assumption that a single backing block alone makes an arbitrary
logical reshape suitable for direct matrix binding; those reshapes now use the
same logical access path.

The existing indexed example composes rank-three concatenation, broadcast
multiplication and scalar addition. CPU and Metal float32 runs complete 2158
submissions, observe the transformed early half with unrelated source/index
regions absent, and complete both generations numerically. The paired Metal
float16 gold workflow also passes; the new rank-three arithmetic remains float32
on rank zero. The peer exits normally after SIGTERM. Compressed logs/traces and
exact installed library/caller revisions are in `rank-pointwise-provenance.json`.
No matched old-emitter timing baseline was collected, and these cases do not
establish all-operator/rank coverage or a throughput improvement.

## Shared scalar/load emitter integration

The scatter agent's `1726082` extracts common recursive scalar expression and
indexed load emission from the ordinary kernel source owner. The existing segment
reducer now shares that recursion while keeping its exact ordinal bounds,
prebound inputs, directory, routing and allocations. Root source review confirms
that pointer arithmetic, dtype casts, numerical masks and reduction references
retain their prior generated expressions. This refactor establishes shared code
for subsequent bounded indexed dependency binding; it does not yet enable
load-valued scatter updates or masks.

Local CPU and Metal float32 workflows pass with 2158 completed submissions each,
and paired Metal float16 passes with 1473 rank-zero completions, unchanged from
the preceding example configuration. Duplicate/masked scatter, delayed chunks,
repeated reuse, logical pointwise/indexed operations, exact integer reductions,
nested/mixed contractions and fanout continue to pass. The peer exits normally
after SIGTERM. `shared-emitter-provenance.json` records raw compressed logs/traces
and revisions. Side cases remain rank-zero local and only gold/fanout traverse
RDMA. No matched performance improvement is claimed.

## Bounded nested scatter loads

`77ad99d` binds load-valued scatter updates through the shared scalar/load
emitter and predicate-aware access traversal. Each selector and numerical
partial evaluates only its segment's explicit ordinal range. Nested coordinate
loads retain those same bounds; direct chunk/panel operands retain their known
Ref identities. Numerical values remain in their original canonical pages.
Selectors contain candidate indices, not copied update rows.

The first operational attempt failed during realization because native active
functions required every selector to share the active-count producer's output.
`9ca899e` replaces that implicit lifetime coupling with explicit indexed reader
memberships on the count maps. Derived selectors publish even for empty ranges;
only numerical partials use active omission. Indexed retirement retains the
domain's omission disposition across subsequent selector resets. `d91b70b`
checks every ordinary metadata dependency during setup, rather than accepting
one rooted path while overlooking another. The shared-memory ABI remains 25;
neither bridge was restarted.

The final installed revision on both machines is `d91b70b`. Local CPU and Metal
float32 examples each complete 2730 submissions. Paired Metal float16 completes
2045 rank-zero submissions. Four scatter generations exercise nested lookup
indices, duplicate destinations, masked contributions, an independently delayed
source chunk, a separately delayed coefficient, an empty routing domain, and
reuse after that empty domain. Unrelated destination consumers finish before the
delayed values are published. Existing gold, logical indexing, integer reduction,
contraction and fanout cases also pass. Only gold and fanout cross RDMA in this
workflow; the scatter side case runs on rank zero. The peer exits zero after
SIGTERM and both bridges remain ready with zero clients.

`bounded-scatter-provenance.json` records exact revisions, compressed logs/traces,
per-generation scatter timings and gold timing count/mean/sample variance. No
matched performance improvement is claimed. Selector capacity remains reserved
per segment/panel at its configured worst-case range, so storage packing and
launch reduction remain work. Load-valued outer routing masks, broader scatter
and derivative caller migration, and the remaining nine-step acceptance criteria
are not established by these examples.

## Runtime scatter masks stay within segments

`937ba01` moves masks containing indexed loads or nonconstant direct inputs out
of routing and into the bounded numerical expression as `select(mask, value, 0)`.
Only declared-constant/static masks retain early routing elimination. This reuses
the shared predicate-aware access collector, so a false mask suppresses dynamic
value loads and nested coordinate loads. Direct mask operands keep their known
Ref readiness granularity. There is no new runtime scheduling interface, native
state, numerical staging buffer or bridge ABI change.

`6d53f11` extends the existing scatter example with runtime mask pages. An earlier
false-masked update points at the delayed final value page; unrelated destinations
still complete with both that value page and the final mask block unpublished.
The empty destination-routing generation finishes while all masks and values
are withheld, then accepts those pages for retirement. The following generation
reuses the same storage successfully. A separately delayed coefficient remains
part of the example.

Final installed library and caller revision `1d3c6fb` passes local CPU and Metal
float32 workflows with 3150 completed submissions each, and paired Metal float16
with 2465 rank-zero completions. The four scatter generations pass exact endpoint
comparisons. Existing gold, contraction, logical-indexing, integer reduction and
fanout cases pass. Only gold/fanout cross actual RDMA; scatter remains a rank-zero
side case. The peer exits zero after SIGTERM. Raw compressed logs/traces, exact
configuration and timing count/mean/sample variance are recorded in
`bounded-mask-provenance.json`.

These dynamic mask selectors add work relative to the preceding constant-mask
fixture; the submission counts are not an equal-work speed comparison. Worst-case
metadata allocation and launch packing remain unresolved performance work.
Broader scatter/derivative caller migration, remaining rank/axis operations and
the full nine-step performance acceptance remain open.

## Take-along-axis transpose caller migration

`0ce49c7` replaces the custom take-along-axis VJP atomic/clear emitter with shared
indexed addition. Forward and backward calls share source-coordinate construction
across logical rank, selected axis and singleton broadcasting. Destination metadata
checks each source axis before flattening, so invalid coordinates cannot alias a
different valid scalar destination. Cotangents are indexed directly from their
original pages. The numerical primal is absent from liveness and peer replication.
Row-aligned scalar output blocks are reframed as matrix views using their original
Tensor/extent/offset identities, preserving publication and downstream bindings.

The existing optional Xonotic example now composes a rank-three broadcast take
transpose with multiplication and addition. Its first output row finishes while
cotangent blocks 1 and 3 remain unpublished. Duplicate valid negative indices,
positive and overly negative out-of-axis indices, and a second generation pass
exact numerical comparisons. No primal input binding is supplied. CPU and Metal
float32 workflows each complete 3238 submissions, and paired Metal float16
completes 2553 rank-zero submissions. Existing streaming scatter masks, gather
transposes, contractions, reductions, gold and fanout cases continue to pass.
The take side case executes in float32 on rank zero; only gold/fanout cross RDMA.
The peer exits zero after SIGTERM and both bridges remain ready with zero clients.

`take-gradient-provenance.json` records installed library/caller revision
`0ce49c7`, raw compressed logs/traces and timing count/mean/sample variance.
These examples establish the recorded broadcast/axis cases, not exhaustive
rank/axis coverage or a matched speedup. Scalar destination metadata and partial
storage still use the existing worst-case scatter capacity.

A concurrent source audit identifies the next caller work: general gather
transposes can share forward gather coordinate construction, per-axis validity,
scalar destination sums and the same shape-only primal handling. Nonvector
scatter needs a semantic correction as well: the old emitter compares row indices
against flat output addresses, whereas Graph.At.add and its gradient[y] derivative
express row indexing. Existing named callers use vector bases and do not expose
that mismatch. General gather and nonvector scatter migration, storage/launch
optimization and remaining nine-step acceptance work remain open.

## General gather transpose caller migration

`d6f5a05` extracts the forward gather's source-coordinate tuple and shares it
with its transpose. The scalar destination, per-axis validity, segmented sum,
original cotangent loads and matrix reframe now use the same lowering block as
take-along-axis gradients. All gather transposes omit numerical primal liveness
and replication. The existing row-gather U×F specialization remains on shared
indexed_add. The separate custom atomic emitter and gather_address generator are
removed; no replacement derivative runtime is introduced.

`ac74558` extends the existing gather-gradient runner with a rank-three primal,
nonadjacent advanced indices, reversed middle-axis slice and a pointwise consumer.
Output regions for source rows 0, 1 and 3 complete while cotangents for source row
2 remain unpublished. Duplicate/negative indices, one invalid per-axis index and
repeated storage reuse pass exact endpoint comparisons. Neither gradient example
supplies a numerical primal binding. Existing row-gather and broadcast-take cases
continue to pass.

Final installed library/caller revision `d6f5a05` passes CPU and Metal float32
workflows with 3406 completed submissions each. Paired Metal float16 completes
2721 rank-zero submissions. Scatter masks, contractions, reductions, gold and
fanout also pass. The advanced gradient side case remains float32 on rank zero;
only gold/fanout traverse RDMA. The peer exits zero after SIGTERM.
`gather-gradient-provenance.json` records compressed logs/traces, revisions,
configuration and timing count/mean/sample variance. These cases are not an
exhaustive advanced-index suite or a matched performance improvement claim.

The general transpose path retains scalar scatter's worst-case metadata and
partial capacity. Nonvector scatter's row-indexing correction is the next caller
migration. Storage/launch optimization, remaining operator coverage and the full
nine-step performance acceptance remain open.

## Row-scatter caller correction and migration

`a80b48f` removes the custom flat-address scatter scan. Xonotic row selection now
uses shared indexed_add for vectors, matrices and canonical higher-rank bases.
Multidimensional index tensors select leading rows; intermediate trailing row
axes expand destination metadata, while the final feature axis remains a vector.
Updates broadcast to the selected logical shape through direct compatible Refs
or logical indexed loads over their actual block table. No expanded numerical
update tensor is created. Noninteger indices and incompatible broadcasting fail
at setup; direct binding no longer assumes any single block can be reshaped.

The existing indexed-sum runner exercises both a matrix base and a rank-three
base with multidimensional indices. Each composes pointwise update production,
row scatter and a pointwise consumer. Nonzero bases, duplicate/negative rows,
feature and higher-rank broadcasting pass exact endpoint comparisons. Rows 0,
1 and 3 finish while the last update block for row 2 remains unpublished, in
both generations. Earlier gather/take gradient cases use the same runner and
continue to pass alongside the other streaming-algebra cases.

Final installed library/caller revision `a80b48f` passes CPU and Metal float32
with 3660 completed submissions each. Paired Metal float16 completes 2975
rank-zero submissions. The scatter side cases are float32 on rank zero;
only gold/fanout cross RDMA. The peer exits zero after SIGTERM.
`row-scatter-provenance.json` records revisions, raw compressed logs/traces,
configuration and timing count/mean/sample variance. This is no matched speedup
claim, and the old nonvector scan was not a correct baseline for these cases.

Base layouts still must admit the existing metadata-only matrix view. Arbitrary
reshape aliases crossing backing partitions remain a logical-layout gap; they
are not copied into alternative dense storage. Broader operator coverage,
shared metadata/partial storage and launch optimization, and full nine-step
performance acceptance remain unfinished.
