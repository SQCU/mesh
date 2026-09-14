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

## Partitioned scatter base views

`b642e24` generalizes matrix_view with regular metadata fragments whose width is
an exact common divisor of target columns, source columns and source column-block
width. Every fragment lies within one source row/backing region; its logical flat
ordinal determines the original Tensor.region slice. Tensor/extent/offset and
strides remain unchanged. `a64eb23` retains a single view where row-major flat
traversal is provably affine, alongside existing equal-shape, vector-transpose
and aligned-column cases. This closes the previous partition-crossing reshape
restriction without a numerical copy or reshape kernel. Aliases retain original
page readiness and cannot be independently published as whole extents.

`4b3d3f7` makes the existing rank-scatter base a runtime 4×6 tensor with 1×2
backing blocks, reshaped logically to 4×2×3. The new mapping therefore crosses
original column partitions. Original base row 2 remains unpublished alongside
the last update block while consumers of rows 0, 1 and 3 complete. Publishing
base row 2 alone does not complete its consumer; the required updates still must
arrive. Final values and the second reuse generation pass exact comparisons.

Final library/caller revision `a64eb23` passes CPU and Metal float32 workflows
with 3796 completed submissions each, and paired Metal float16 with 3111
rank-zero completions. Existing gather/take gradients, matrix scatter, masked
scatter, contractions, reductions and fanout continue to pass. The reshape side
case runs in float32 on rank zero; only gold/fanout cross RDMA. The peer exits
zero after SIGTERM. `scatter-base-view-provenance.json` records compressed raw
logs/traces, revisions, configuration and timing count/mean/sample variance.

The no-copy conclusion comes from the source's metadata slice/view construction,
not from timings. A width gcd of one can require one metadata entry per scalar
and can shrink scatter's numerical output regions. This explicit setup/launch
cost remains optimization work; these examples do not establish exhaustive layout
coverage or a matched speedup. The full nine-step performance and operator
coverage requirements remain open.

## Eliminate identity and unused range producers

`64c9246` removes derived flattened-range allocation/production for scalar panels
and direct-only update expressions. Width-one flattening is exactly [lo, hi),
and direct-only expressions have no flattened selector-range consumer. Existing
directory views and indexed-reader memberships preserve the required lifetime.
Wider panels containing indexed loads retain their scaled bounds. An independent
source audit confirmed nested update/mask loads are included in indexed_inputs;
rewritten direct Ref and ordinal loads cannot introduce untracked Tensor selectors.

The example source and configuration are unchanged from `a64eb23`. CPU and Metal
float32 traces each fall from 2131 to 2095 configured functions and from 3796 to
3724 completed submissions. Paired Metal float16 falls from 2067 to 2031 configured
functions and from 3111 to 3039 rank-zero submissions. All recorded numerical
comparisons, independently withheld masks/base fragments/updates/cotangents,
consumer progress and repeated reuse continue to pass. The paired gold/fanout
work crosses actual RDMA; side cases remain on rank zero. The peer exits zero
after SIGTERM.

`range-elision-provenance.json` retains exact library/caller/baseline revisions,
compressed raw logs/traces, function/submission counts and timing count/mean/sample
variance. This demonstrates 36 removed configured range producers and 72 fewer
submissions in this workflow, not a matched wall-time speedup. Larger storage and
launch costs remain: all-constant indexed tables need no flat selector range,
and equal-width feature panels can sometimes share selectors when expression,
bounds and referenced layouts all match. Sharing must retain those identities
and lifetime fanout rather than introducing new synchronization. Broader operator
coverage and the full nine-step performance acceptance remain open.

## Share bounded selector common subexpressions across feature panels

`ba94e38` shares identical selectors across equal-width feature panels within one
scatter segment. The setup key preserves the expression, referenced input layouts
and page identities, constantness, ordinal and bound views, active count and
segment width. Only inputs actually used by the selector contribute to its key.
Row-only selectors therefore share without conflating panel-dependent addresses.
The native reader fanout preserves every consumer's lifetime; numerical operands
remain the original registered pages. Constant-only indexed expressions also omit
unused flattened-range producers. No invocation scheduler or synchronization was
added.

The unchanged example/configuration from `a64eb23`, compared with library
`64c9246`, passes CPU and Metal float32 and paired Metal float16 numerical,
withheld-input progress and repeated-reuse checks. Local traces fall from 2095
to 2071 configured functions and from 3724 to 3676 completed submissions. Paired
rank-zero traces fall from 2031 to 2007 functions and from 3039 to 2991 submissions.
Thus each workflow removes 24 configured functions and 48 submissions. Paired
gold/fanout cross RDMA; indexed side cases run on rank zero. The peer exits zero
after SIGTERM, and both bridges remain ready with zero clients.

`measurements/lowering-2026-09-13/selector-cse-provenance.json` records source
revisions, raw compressed logs/traces and timing count, mean and sample variance.
These three-invocation measurements establish removed work and preserved progress,
not a matched wall-time speedup. Selector capacity per unique expression and
fragment-driven launch costs remain; expert/neighborhood caller migration and
full nine-step performance acceptance are still open.

## Migrate neighborhood forward and all derivatives

`e1682a8` replaces the whole-region neighborhood emitter with shared indexed
loads, FP32 edge products and row reductions, and bounded indexed-add. The
forward observer sum and query/key/value/weight derivatives use the same region
owner. `numerical_operands` supplies one target-specific dependency list to both
liveness and peer replication; nongram query/key gradients are constant zero
pages. The old neighborhood emitter, launch modes and unused atomic helper are
deleted. The algebra and prior art are in algorithm-sources.md under Xonotic
neighborhood algebra. Independent source review checked operand slots after two
statistic appends, original-edge ordinals, feature offsets, duplicate sums and
active-domain lifetimes.

`4fe038b` extends the existing streaming-algebra example with both Gram modes,
all four derivatives and pointwise consumers. Four observers each have two
neighbors and three features. Duplicate indices change between two generations.
Query/key/value/cotangent row2 remains unpublished while rows0/1/3 consumers
finish. Nongram query/key consumers finish on every row, including row2, without
payload dependencies. All forward/derivative results agree with float64 reference
algebra at rtol/atol2e-5, before and after reuse.

CPU and Metal local workflows pass, as does paired Metal with the existing
float16 FFN gold/fanout crossing RDMA. Neighborhood cases themselves use float32
on rank zero, so cross-peer neighborhood execution and half-precision neighborhood
coverage remain unmeasured. The peer exits zero after SIGTERM; both bridges
remain ready. Compressed logs/traces, revisions and per-mode early/completion
count, mean and sample variance are in
measurements/lowering-2026-09-13/neighborhood-provenance.json.

The expanded fixture configures3129 functions and completes6040 submissions
locally; paired rank zero configures3065 and completes5355. These totals include
the newly added neighborhood programs and cannot be compared as a speedup with
the prior smaller fixture. There is no matched timing baseline for the removed
whole-region neighborhood implementation. Explicit E×D FP32 product storage,
duplicate statistics across separately requested derivatives, and segmented-sum
metadata/launch expansion remain substantial shared-lowering optimization work.
This closes the neighborhood caller's whole-region binding gap, not the full
nine-step performance acceptance. Expert routing/contractions and remaining
operator coverage still require migration and measurements.

## Share repeated neighborhood statistics

`b12442a` retains one raw FP32 Q·K or G·V edge statistic for matching users
within one setup realization. The key preserves operand/index graph identities,
participant and numerical domain/tile dimensions; immutable realized bindings
and the existing replica map supply their physical identity. Each user's own
expression slot refers to the shared result. Independent source review confirmed
alias handling, argument slots, liveness and native reader fanout.

The unchanged neighborhood workflow passes CPU, Metal and paired Metal, including
both modes, all derivatives, duplicate indices, withheld-row progress and reuse.
Local configured functions fall from 3129 to 2969 and submissions from 6040 to
5656. Paired rank-zero functions fall from 3065 to 2905 and submissions from 5355
to 4971. Each workflow therefore removes 160 configured functions and 384
submissions. Raw traces/logs and timing count, mean and sample variance are in
measurements/lowering-2026-09-13/neighborhood-cse-provenance.json. This is evidence
of removed work, not a matched wall-time speedup. Neighborhood checks remain
float32 on rank zero; paired gold/fanout traverse RDMA.

The shared expression source still assigns scalar coordinates shape (1,1).
Indexed loads derive their domain from those coordinate expressions, so scalar
loads alone cannot specify the width of an indexed reduction. Retaining vector
index lengths and tile extents explicitly is the next library requirement for
fusing edge products into reductions without temporary E×D product tensors.
That fusion, expert migration and full nine-step performance acceptance remain
open. No application-specific fused kernel was added to bypass the missing
representation.

## Retain index-vector domains and fuse indexed reductions

`8352bb4`, `3cb621a` and `0e4244a` add `kernels.arange(length, tile=...)` as an
integer expression carrying its length, tile and offset. It allocates no index
operand. Reduction lowering retains that domain through loads and arithmetic,
then emits bounded indexed partial sums. Static access pruning, runtime selector
lengths and scalar emission use the same retained extent. Singleton broadcasting
preserves the offset, including a length-one ragged tail. Scatter expressions
resolve feature vectors through the existing feature-domain lowering; vector
masks apply to numerical contributions rather than scalar routing.

`4bce209` replaces neighborhood's materialized E×D product tensors with one
indexed product-and-sum expression. Existing FP32 scalar partials and their
reduction remain. `9abf63c` first changes the existing neighborhood fixture to
width3/tile2 and records a passing Metal baseline. `8a65cdf` also uses explicit
vectors in its existing gather and scatter expressions, preserving their numeric
work. Independent source review checked tile offsets, selector domains, static
pruning, dtype, cache identities and singleton behavior.

CPU and Metal local workflows pass, as does paired Metal. The neighborhood
cases still run float32 on rank zero and cover all derivatives, duplicate indices,
withheld source/cotangent row progress and reuse. Existing gather/scatter cases
exercise ordinary vector use, nested dynamic loads, masks and empty/reused
occurrences. Paired float16 gold/fanout use actual RDMA. Local traces contain
3449 configured functions and 6712 submissions; paired rank zero has3385 and
6027. The comparable ragged Metal baseline contains3497 and6808: fusion removes
48 configured functions and96 submissions. This is removed-work evidence, not
a matched wall-time speedup claim.

Raw baseline/current logs and traces, revisions and timing count/mean/sample
variance are in measurements/lowering-2026-09-13/index-domain-provenance.json.
The source proves removal of full edge-product storage; scalar partials,
selectors and launches remain. The public API has a complete indexed-reduction
example in algorithm-sources.md. Broader vector-axis/shape coverage, expert
migration, cross-peer neighborhood acceptance and matched end-to-end performance
remain part of the full nine-step objective.

## Matrix-shaped index domains

Source review of expert contractions identified independent output-feature and
contraction dimensions as the next representation requirement. `4a31ee6`
extends the existing index vector with an axis: `.T` flips that metadata without
allocating a coordinate tensor or a transpose kernel. Shared layout, narrowing,
static access specialization, selectors and scalar emission retain the axis.
In scatter updates, column vectors refer to original update ordinals across
chunks, and row vectors refer to features. Nonsingleton vector masks on either
axis use contribution masking, not chunk-local scalar routing. Independent
review confirmed these paths and withdrew an initial incorrect mask concern
after rereading the current runtime_mask source.

The existing mapped contraction now expresses an indexed product-and-sum with
broadcast row and column vectors, preserving its reversed source BlockSpec
mapping. The streamed gather also uses broadcast vectors, and scatter uses a
full update-row vector. Initial CPU/Metal runs at `8d1be65` pass. `53f5b63`
then tiles seven output features by three while reducing five contraction
features by three, covering ragged tails on both axes. Final CPU, Metal and
paired Metal runs pass numerical comparisons, withheld-input progress and reuse.
The paired run includes float16 inputs in the indexed matrix expression; its
neighborhood cases remain float32. Gold/fanout traverse RDMA; the indexed side
cases run on rank zero.

Raw logs/traces and exact revisions are recorded in
measurements/lowering-2026-09-13/index-axes-provenance.json, including timing
count/mean/sample variance. Final local traces configure 3467 functions and
complete 6760 submissions; paired rank-zero counts are recorded in that artifact.
These are coverage measurements, not speedup evidence. The indexed matrix
example is not a replacement for the optimized matmul backend used by serving
or the gold chain. Expert routing, forward/input/weight derivative caller
migration and performance acceptance remain open; the matrix-shaped index
representation needed to express that work is now available and exercised.


## Expert indexed contractions and independent derivatives

`30e690b` migrates expert forward, input derivative and weight derivative to
shared indexed products, reductions and indexed_add. It deletes the custom
expert route and numerical emitter. Input derivatives do not bind input values;
weight derivatives do not bind weights. Registered source views and canonical
indexed-reader lifetimes remain the execution operands.

The existing operational example (`cbee5db`) covers ragged feature/contraction
tiles, duplicate selections, an empty expert and changed selections over two
generations. Independent experts and their downstream pointwise consumers finish
while expert 2 payloads are withheld. In generation zero, expert 2's weight
derivative finishes before its weights arrive. In generation one, its input
derivative finishes before its input arrives. Every completed forward and
derivative consumer matches the numerical reference.

CPU, Metal and paired Metal runs pass. Expert cases are float32 on rank zero;
the paired float16 gold chain and fanout traverse RDMA. Local traces configure
4064 functions and complete 8050 submissions; paired rank zero configures 4000
and completes 7365. Raw logs, traces, revisions and timing count/mean/sample
variance are in measurements/lowering-2026-09-13/expert-provenance.json.

The initial run exhausted the fixed 32768-page arena during setup. The failure
log is retained. Healthy idle bridges were gracefully resized through the
existing bridge command to 65536 pages, four pages per block and one QP on both
nodes: 1097203712 registered bytes per node. The existing visible memory
preflight remains in place. This is configured arena capacity, not a claim of
live operand bytes or system memory exhaustion.

These runs establish numerical behavior and independent publication, not a
speedup. Scalar indexed expert reductions still require comparison with a tuned
grouped contraction implementation. Batched contractions, higher-rank reductions,
shared contraction optimization and matched end-to-end performance acceptance
remain open under the nine-step plan.


## Batched native contractions and ranked reductions

`c52c152` maps real batched matmul to the existing shared dot / nn.linear
lowering. Broadcast-prefix ordinals and transpose views are realized at setup.
Matrix batch views retain source Ref identities, offsets and strides; combining
output batches composes their Ref tables without copying or republishing data.
Output row tiles divide matrix rows so batch boundaries remain regular blocks.
Independent source review checked broadcast alignment, ragged backing boundaries
and native allocation lifetime. CPU float32 contractions retain BLAS; Metal
contractions retain MPS. This is source-level reuse, not a matched speedup claim.

The same increment expresses higher-rank sum/mean through broadcast index
vectors and shared reductions, retaining logical axes and keepdims. This includes
the batch sum required by a singleton broadcast weight's gradient. Separate
output and input-gradient regions do not acquire that all-batch dependency.
Independent source review checked coordinate mapping, tails, integer narrowing
and selected region dependencies. Integer matmul remains outside this migration.

`dd82efc` extends the existing operational workflow with both-transpose
(2,5,3) by (1,7,5) matmul, ragged tiles, both derivatives and pointwise consumers.
Two generations alternate which batch arrives first. The late input derivative
can complete from its cotangent and weights while its primal remains absent;
the forward result and shared weight derivative still require that primal.

Remaining reachable caller work includes boolean pointwise masks and broadcast,
max/min/any reductions, sorting and unshared scalar/generation operations.
The nine-step feature and matched performance acceptance remains open.


CPU, Metal and paired Metal operational runs at c52c152 pass numerical and
withheld-input checks, including both generations of the new batched case.
Rank-zero side cases use float32; paired gold/fanout use float16 over RDMA.
Local traces configure 5168 functions and complete 10258 submissions. Paired
rank zero configures 5104 and completes 9573 submissions. The expanded case
adds work; these totals are not matched performance comparisons. The local
Metal first generation is visibly slower than its second, and no warmup or
speedup inference is made from two observations. Raw logs, traces, per-generation
events and timing count/mean/sample variance are retained in
measurements/lowering-2026-09-13/batched-provenance.json. The broadcast VJP
exercises higher-rank sum; other higher-rank sum/mean axes have source review
rather than operational coverage in this increment. Both bridge arenas retain
the previously configured 65536-page capacity; peer teardown exits zero.


## Boolean masks and explicit broadcasting

`edf7d4b` routes logical not/and/or and broadcast through the existing shared
pointwise expression lowering and removes their obsolete emitter cases.
Numerical truth normalizes through equality-to-zero, then boolean combination;
NaN is true and either signed zero is false. Both numerical operand dependencies
remain explicit. Rank-two zero-stride views and higher-rank logical coordinates
retain original registered storage. Explicit result allocation remains visible;
unobserved broadcast-result fusion is still shared compiler work.

The existing workflow at `f74b265` covers both direct rank-two and higher-rank
paths, numeric NaN/zero/nonzero values, scalar broadcasts, pointwise consumers,
an independently withheld source row and changed values across reuse. Independent
source review found no defect in truth conversion or valid broadcast mapping.

## Next native indexed contraction boundary

Source review at f74b265 identifies a required performance step beyond caller
migration. Expert forward/dX expressions are indexed product-and-sum contractions,
but shared panel lowering materializes computed operands and native bind_part
fixes CPU addresses and MPS matrices at setup. Indexed reader registration tracks
readiness and lifetime; it does not redirect those fixed numerical bindings.
Merely rewriting the expression to dot therefore does not recover a native
selected-panel contraction without introducing operand staging.

The next shared implementation should retain a selector and exact candidate
matrix views before logical flattening, prepare backend bindings at setup, and
register one canonical output producer that indexes those retained bindings.
Numerical-plan identity and readiness identity must remain separate: different
expert subviews can occupy the same backing page, while indexed readiness
candidates must not overlap. Retain plan-to-original-page mappings and deduplicate
only readiness identities. Tile boundaries must keep selected panels representable
by native strides. General nonaffine row gathers remain an indexed-kernel problem;
one selected affine panel is the first concrete native contraction case.

This is a source-derived implementation boundary, not a completed optimization.
Prior art is Pallas scalar-prefetch block indexing and its explicit grouped/ragged
contractions; see https://docs.jax.dev/en/latest/pallas/tpu/sparse.html and
https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py.
No additional execution owner or model interpretation belongs in transport.


Boolean/broadcast CPU, Metal and paired Metal runs at f74b265 pass both
generations of the existing numerical/progress workflow. Local traces configure
5266 functions and complete 10472 submissions; paired rank zero configures 5202
and completes 9787. Rank-zero boolean cases use float32 numeric inputs, and the
paired gold/fanout use float16 over RDMA. Raw events and compute traces are in
measurements/lowering-2026-09-13/boolean-provenance.json, alongside full workflow
timing count/mean/sample variance. No boolean-specific host timing or matched
speedup claim is made. Both participant processes exit zero and registered arena
geometry remains unchanged. The nine-step goal remains active.


## Prepared selected native contractions

`880e01f` factors backend preparation from ordinary contraction registration and
adds selected contractions with one canonical output producer per publication
quantum. Candidate CPU/MPS/Core ML bindings are prepared at setup; numerical
execution indexes a prepared binding and forwards completion to the parent.
`243b947` exposes touched-page views through the existing native dependency
mapper, avoiding a second stride-to-page algorithm in Python.

`e49e292` recognizes affine selected logical product-and-sum expressions before
logical coordinate flattening. Exact numerical views and disjoint readiness page
identities remain separate. K/feature boundaries are refined across candidates;
shared FP32 reductions and required output assembly preserve original publication.
Selector expressions and constant-zero panels reuse setup bindings. Invalid
selection contracts the real operand with a zero panel, preserving NaN behavior.
Root and independent reviews checked native completion ownership, page-map
coverage, alias deduplication, transpose strides and selector lifetime. The native
library builds successfully. No bridge/dataflow structure ABI changes were made.

The existing expert example at `8e7d0c7` packs weights so expert0/1 share a
backing block, includes negative and invalid indices and a NaN input, and retains
the delayed expert2 and independent derivative checks. Scalar Metal baselines
are archived in measurements/lowering-2026-09-13/selected-baseline-provenance.json.
The selected implementation was installed from `80076b1` on both nodes; source
hashes of the local installed Python implementation match the worktree.

Local Metal passes the same strengthened workflow. It configures 5258 functions
and completes 10338 submissions, versus scalar baseline 5266 and10442. These
counts demonstrate removed work, not a statistically established speedup. General
nonaffine/grouped contractions, larger-than-quantum selected root lowering,
remaining caller operations and matched performance acceptance remain open.
Core ML preparation reuse is source-reviewed; this increment does not establish
ANE placement or internal zero-copy behavior through a Core ML measurement.


Final CPU, Metal and paired Metal runs pass at 80076b1. CPU/Metal each configure
5258 functions and complete 10338 submissions; paired rank zero configures 5194
and completes 9653. Raw logs, traces, exact implementation revisions and expert
count/mean/sample variance are in
measurements/lowering-2026-09-13/selected-native-provenance.json. The installed
native library and compiler are synchronized to both nodes; all participant
processes exit zero, with ordinary SIGTERM peer teardown. The registered arena
remains 65536 pages, four pages per block and one QP per node. This closes the
selected affine-panel implementation increment, not the full nine-step goal.

## Shared extrema and truth reductions

`8a1a9ae` extends the existing expression/region reduction owner to max, min,
any and all, sharing sum's axis handling, indexed dependencies and partial tree.
`b25b6dd` moves every Xonotic reduction to those expressions and deletes the
whole-region reduction emitter and dispatch mode. Empty-axis numerical identities
and boolean truth conversion remain explicit. The compiler retains integer
extrema without FP32 conversion; Metal's 64-bit extrema use ordered high/low
32-bit SIMD reductions. Nested non-sum reductions retain the operand's inferred
accumulation type. Numerical invocation adds no scheduler, allocation or binding.

The numerical contract and literature are in algorithm-sources.md under
shared-associative-reductions. Floating extrema preserve initialized numeric
fmax/fmin behavior, including all-NaN infinity identities. This is deliberately
documented separately from JAX's NaN-propagating extrema semantics.

`9c59636` extends the existing streaming-algebra Xonotic workflow with 24 cases:
ranks one through three, exact signed/unsigned 64-bit extrema, fractional/NaN/
signed-zero truth, floating NaN extrema, ragged columns, downstream consumers,
alternately withheld independent rows and two generations of storage reuse.
`177d9a2` records consumer timing in the same workflow. Source and generated CPU
syntax checks pass. The integrated Metal run at `8a1a9ae` passes all cases,
configuring 5566 functions and completing 10954 submissions with runtime code0.
This enlarged workflow is not a matched performance baseline for the old emitter;
no speedup or performance parity claim follows from those counts.

Independent source review found a composed typing defect not exposed by the
individual reductions: `(float_x.sum() > 0).any()` inherited boolean accumulation
context and truncated fractions. `ebcd5e6` corrects operand type propagation
through expressions generally. `7da6a30` adds this direct shared expression and
its all variant to the same workflow. The strengthened Metal workflow fails
with library `8a1a9ae` at the early numerical comparison and passes with
`ebcd5e6`. It configures 5590 functions and completes 11002 submissions. The
paired Metal workflow also passes, with 5526 functions and 10317 submissions;
peer teardown uses ordinary SIGTERM and exits zero. The 26 reduction cases run
locally on rank zero; the composed gold and fanout exercise the RDMA link.

The final CPU workflow at `ebcd5e6` also passes all 26 cases, configuring 5590
functions and completing 11002 submissions with runtime code0. The initial
24-case CPU run at `8a1a9ae` passed as well; the new composition was needed to
expose the source-reviewed typing defect. All successful runs, the pre-fix
failure, source revisions, count/mean/sample variance, and raw compressed logs
and traces are recorded in
[associative-provenance.json](../measurements/lowering-2026-09-13/associative-provenance.json).
The arena remains 65536 pages, four pages per block and one QP per node.

The next smallest caller migration is the shared scalar intrinsic family:
arcsinh, logaddexp, expm1 and isfinite. Preserve near-zero and large-input
behavior, infinity/NaN semantics and exact integer predicates with numerical
primary sources. Generators, sorting, integer contractions, broader indexed
native contraction recognition and the nine-step performance acceptance remain
open. The removed reduction emitter does not imply those other paths are gone.

## Shared elementary numerical functions

`ad94c97` adds elementary expressions and CPU/Metal scalar lowering to the
existing region owner. Stable log1p/expm1/asinh/logaddexp, classification, abs,
log/sqrt/power and corrected floor division share original source-page bindings.
`af4bc73` selects precise Metal remainder explicitly, alongside precise variants
for new domain-sensitive functions. Safe arithmetic and the default math-function
family are independent SDK settings; existing exp/tanh and global compile settings
remain unchanged. Mechanisms and numerical boundaries are documented in
algorithm-sources.md under shared-elementary-functions.

`9da0250` migrates the entire remaining public Xonotic elementary family, including
integer inversion and floor division, through the shared pointwise/logical-index
owner. The caller's generic element emitter, four numerical helper functions and
obsolete broadcast/coordinate helpers are deleted. Only dimension/arange/random
normal, sorting and integer contractions remain in that custom source emitter.
The migration corrects old log1p(+Inf), logaddexp NaN and truncating integer
floor-division behavior; it does not claim exact integer exponentiation.

`153c6b7` extends the existing workflow with near-zero and large values, NaNs and
infinities, signed zeros, integer endpoints, downstream operations, ragged regions,
independent withheld rows and repeated use. `128ba81` retains actual and expected
values when a comparison fails. That diagnostic and NumPy source inspection
identified a reference discrepancy: scalar exponent 0.5 was rewritten to sqrt.
`6efcb83` uses generic NumPy power instead. The precise Metal implementation was
already correct for this case and was not replaced with a special-case helper.

Final Metal and paired Metal pass with library `af4bc73` and example `6efcb83`.
Local Metal configures 5830 functions and completes 11482 submissions; paired
rank zero configures 5766 and completes 10797. Both report runtime code 0, and the
peer exits zero after ordinary SIGTERM teardown. New elementary cases run locally
on rank zero; the gold chain and fanout traverse RDMA. These expanded workflows
have no matched old-emitter timing baseline, so no speedup or performance parity
claim is made. Full nine-step performance acceptance remains open.

The final CPU run also passes, with 5830 functions and 11482 completed submissions,
runtime code 0. All 19 elementary cases pass both generations. Raw logs/traces,
source revisions, reference-discrepancy evidence and count/mean/sample variance
are in [elementary-provenance.json](../measurements/lowering-2026-09-13/elementary-provenance.json).
The registered arena remains unchanged and both bridges return idle and ready.

The next source-reviewed migration is setup-resolved dimensions and indexed range
production. Existing typed indices and expressions suffice for generated values;
negative-step lengths and empty-range representation require explicit graph work.
After that, shared counter-based Philox/Box–Muller, typed stable sorting and integer
contraction semantics still need implementation. Static folding, fusion, placement
and matched performance acceptance remain part of the original nine-step goal.

## Realize symbolic dimensions and stream typed ranges

`43ca4c4` moves Xonotic dimensions to existing setup constants and ranges to
shared tiled expressions. Signed/unsigned integer arithmetic retains all 64 bits;
exact signed ceiling division resolves range length before numerical execution.
The separate dimension/range source paths, runtime dimension renderer and
duplicate dimension argument buffer are removed. Remaining custom operations
are normal generation, ordering and integer contractions.

`2e4c7ec` gives empty tensors metadata without allocating native operand storage.
Broadcasting retains zero dimensions, expression output pruning removes absent
work, and logical empty domains retain shape/type while avoiding meaningless
zero-divisor indexing. Shared reductions with nonempty outputs emit identities;
zero-inner-dimension contractions emit zeros. Integer/bool means now return FP32
rather than truncating back to their source type; empty means use floating NaN.
The [mechanism and primary sources](algorithm-sources.md#indexed-range-generation)
describe these setup decisions and numerical contracts.

`464659a` and `43c0fcd` extend the existing Xonotic workflow with five exact integer
ranges, one FP16 range computed with FP32 arithmetic, symbolic dimension
composition, three empty ranges, six empty reductions apiece, and a fractional
integer mean. Each of two generations publishes one input row, observes its
consumer while the other row remains absent, then releases the other row. The
mean correctly retains its dependency on both rows. No new evaluator is added.

Custom native/Metal kernels with empty operand bindings, mixed empty custom
outputs and positive-base indexed-add with zero updates remain unfinished. The
source-level zero-inner-dimension contraction handling is not separately exercised
by these new range cases. General indexed native contractions, generator/order
migration, legal fusion/storage/placement and matched full performance acceptance
remain part of the active nine-point plan.

Source review also traces the retained custom binding after dimension-buffer
removal: addresses occupy slot 0, Views 1, Blocks 2 and argument IDs 3; native
constant binding preserves that order. Concrete fallback shapes preserve tensor
indices/dtypes and supply integer batch-stride products. Those remaining paths
are not numerically covered by the range example. Their pre-existing limitations
remain work: integer contractions convert through FP32 tiles, and argsort's
comparison-based ranks do not define collision-free NaN ordering.

Operational evidence in
[`range-generation-provenance.json`](../measurements/lowering-2026-09-13/range-generation-provenance.json)
records installed/source `2e4c7ec` on both nodes. CPU and Metal each configured
5934 functions and completed 11717 submissions; the paired Metal workflow
configured 5870 functions and completed 11032 rank-zero submissions. All terminal
runtime codes are zero. The peer exited zero after SIGTERM, and both bridges
returned to ready, unpaired, zero-client state with their existing registered
arena geometry. The range side cases execute on rank zero; paired gold/fanout
uses the actual RDMA link.

Empty range reductions returned zero, false, true, negative infinity, positive
infinity and NaN as specified. Nonempty integer means preserved 4.75 and 5.75
after the downstream quarter addition, while remaining absent until both source
rows were published. All six ranges and their independent row consumers passed
across both reuse generations. Early-consumer elapsed times have count 2 each:
CPU mean 0.3673545 ms, sample variance 0.0018886043405 ms²; Metal mean
0.720354 ms, variance 0.004929450632 ms²; paired-workflow rank-zero mean
0.883979 ms, variance 0.022693446882 ms². These are host observation intervals,
not physical compute/wire overlap durations or a matched speedup measurement.
Raw logs/traces and full commands are retained with the provenance.

## Share counter-based random generation

`e48bbaf` adds Philox4x32-10 word expressions and indexed normal expressions to
the shared numerical library. One compact integer helper implements the fixed
rounds for CPU and Metal; the normal transform uses the prior ordinal/key
mapping and precise Metal scalar functions. `ce7a508` replaces Xonotic's
whole-output random binding with ordinary tiled expressions and deletes its
private Philox source. Output region ordinals retain the global flattened
index, including pairs split across row/tile boundaries. No sequence cursor,
host-generated random operand, extra scheduler or runtime allocation is added.

`df8bd3e` extends the existing workflow with three published Philox vectors,
two normal key generations and independent downstream consumers. A second
producer→consumer chain starts at ordinal 2^33 and consumes separate per-row
key pages; one chain completes while the other key remains unpublished. The
normal reference retains exact integer rounds and FP32 uniform conversion;
backend transcendental results use a finite numerical tolerance.
[Mechanism and primary sources](algorithm-sources.md#counter-based-random-generation)
separate exact integer output guarantees from normal floating precision.

Ordering and integer contractions remain in the caller's custom emitter.
Independent Philox word outputs can repeat integer helper work; cross-output
fusion, broad statistical evaluation and a matched fastest-backend performance
comparison remain open. This migration is progress within the nine-point plan,
not completion of its performance acceptance.

The next source disposition is explicit. Integer dot currently hardcodes FP32
accumulation in the shared region owner, while the caller's remaining integer
matmul casts inputs into FP32 tiles. Extend the shared dot owner with typed
modular panel arithmetic, typed merges and publication, preserving the existing
floating native path. Boolean contraction needs its truth-semiring contract.
Sorting should use typed stable runs and configured merges in the library,
retaining original ordinals and NaN ordering. Final ranks depend on their entire
sorted axis; partial runs and unrelated rows can progress independently. These
are implementation tasks, not capabilities supplied by the random migration.

[`random-generation-provenance.json`](../measurements/lowering-2026-09-13/random-generation-provenance.json)
records source and installed library `e48bbaf` on both nodes. CPU and Metal
each configured 5962 functions and completed 11773 submissions. The paired
Metal workflow configured 5898 functions and completed 11088 rank-zero
submissions. All terminal runtime codes and process exit codes are zero; the
peer exited after SIGTERM. Both bridges returned to ready, unpaired, zero-client
state with the existing arena geometry. Random side cases execute on rank zero;
paired gold/fanout uses the actual RDMA link.

All three integer vectors match exactly. Normal values and downstream arithmetic
pass the recorded tolerances across ragged pair boundaries and reuse, including
ordinals whose counter has a nonzero high word. An available per-row key drives
its separate normal producer and pointwise consumer to completion before the
other key arrives. The Xonotic chain separately demonstrates consumers progressing
with the other consumer input row absent.

Early observation times have count 2 each: CPU mean 0.4298125 ms, sample variance
0.0008802788405 ms²; Metal mean 0.820188 ms, variance 0.015745670882 ms²; paired
workflow rank-zero mean 0.7445 ms, variance 0.00114003125 ms². Full observations,
commands and traces are retained. These host intervals are neither a physical
compute/wire occupancy measurement nor a matched speedup comparison.

## Share exact integer and boolean contractions

`a64c782` extends the existing dot region owner with typed integer panels and
boolean truth accumulation. Unsigned arithmetic precedes every modular product
and sum; typed partial merges also apply inside a composed expression. Empty K
produces the correct zero/false identity. Integral operands in a floating dot
use cached numerical cast panels, and output conversion uses a typed expression
when native floating affine cannot represent the requested type. Existing
floating native contractions retain their implementation.

`103e4d0` sends every represented Xonotic rank>=2 matmul through shared batched
views and `nn.linear`, retaining the graph's declared output dtype. It deletes
the custom FP32 integer contraction, its threadgroup tile source, launch
geometry and dead clear machinery. Ordering is the sole remaining custom
numerical source path. The [mechanism](algorithm-sources.md#typed-integer-contractions)
links the arithmetic and panel decomposition to their primary sources.

`1608fd9`, `5ca7d30` and `d6ac83a` extend the existing workflow with I32/U32/I64/U64
and boolean contractions. Values cross FP32/FP64 exact-integer limits and
exercise modular overflow. Both transpose flags, singleton batch broadcasting,
ragged K panels, a second contraction, pointwise consumers and repeated reuse
are included. The direct I64 expression checks a dot epilogue with two partials.
Empty K identities and in-range mixed arithmetic/output conversions are checked;
boolean weights admit multiple true contributions across separate K panels.
An absent batch remains independent of a ready batch's full composed chain.

The integer panel kernel currently maps output columns across lanes and uses
a serial local K loop over actual source strides. This preserves panel/row
parallelism and exact arithmetic, but is not optimized integer microtiling or
a fastest-backend performance result. General indexing, storage/fusion/placement
and matched full-plan performance acceptance remain open.

[`integer-dot-provenance.json`](../measurements/lowering-2026-09-13/integer-dot-provenance.json)
records source `d6ac83a`, installed source `245f4dc` and library `a64c782` on both
nodes. CPU and Metal each configured 6383 functions and completed 12609
submissions. The paired Metal workflow configured 6319 functions and completed
11924 rank-zero submissions. Every runtime/process terminal code is zero; the
peer exited zero after SIGTERM. Both bridges returned to ready, unpaired,
zero-client state with their existing registered arena geometry.

All five typed contraction chains match their exact references across both
reuse generations, including empty K, boolean multiple contributions and the
direct I64 dot epilogue. Mixed I32→FP32 arithmetic→I32 publication returns one;
the floating-output case returns 1.25. A complete early batch's contraction
chain is observed while the other batch remains unpublished. Raw observations,
traces and per-dtype timing count/mean/sample variance are retained.

The integer side cases execute on rank zero; paired gold/fanout crosses actual
RDMA. New integer numerical kernels are not separately measured on rank one.
These host observation intervals establish the covered progress/correctness
behavior, not physical wire/GPU overlap duration or a matched speedup.

## Complete shared ordering and preserve indexed view mappings

`4dc6322` introduces shared stable U32 argsort using bounded Batcher ordinal runs
and tiled two-level Merge Path. Key values retain their original typed canonical
Refs; every merge retains the actual key readers. Independent source runs can
complete before the sort axis is fully available. Final ordering depends on its
own axis, while unrelated rows remain independent. `74725c3` migrates Xonotic
argsort/argpartition, retaining axis and resolved kth semantics, and removes its
last private numerical Metal emitter and binding machinery. Full stable sort
satisfies partition semantics but is not an optimal-selection performance claim.
The [ordering mechanism](algorithm-sources.md#stable-indexed-ordering) cites its
primary numerical algorithms and API contracts.

Source review also found that rank-two transpose unnecessarily emitted a gather
copy. `46510f9` preserves it as actual transposed Ref metadata and retains source
ownership. `e974482` adds canonical full-extent replication with exact alias
remapping; `ead87c6` removes the caller's replica cache and first-block layout
guess. Each distinct original backing extent has one route per peer pair, and
every destination alias retains its original offset, shape and strides. There
is no host numerical copy or alternate dense operand store. The
[replication mechanism](algorithm-sources.md#canonical-view-replication) states
its physical publication unit and lifetime.

A larger fragmented key panel exposed a prior static-gather source problem:
the specializer computed its exact candidate set, then expanded it into deeply
nested conditionals. `59ce724` retains the original candidate ordinal/Ref vector
and emits indexed loads with static reader bindings. Ordering output assembly
uses the same representation across backing tiles. `f53538a` similarly preserves
the variable-width result interval directory in selected-contraction assembly.
The [static specialization mechanism](algorithm-sources.md#static-indexed-access-specialization)
describes the retained data, masks and dependency proof. Compiler bracket limits
are unchanged; the 131-fragment generated CPU sources compile at default limits.

`6326142`, `cc13713`, `90e399e`, `04c2039` and `90e6b92` extend the existing
streaming-algebra workflow with F32/F16/bool/I64/U64 stable ordering, NaNs, ties,
signed zeros, exact large integers, rank-one/rank-two/rank-three axes, flattened
ordering, negative kth, top-k gathers and a 131-element ragged case. Both source
partial work and full independent-row completion are observed across two reuse
generations. A fragmented reshape/transpose also feeds arithmetic on the peer
and returns independently usable output columns, with another source extent
withheld. The local variant exercises the same numerical views.

Three setup failures are preserved: an excessive one-element fixture tile grid
exhausted the fixed arena; a static-gather conditional expansion exceeded Metal's
bracket depth; and the new alias fixture initially bound a plain pointwise input
as a whole indexed tensor rather than its explicit input region. The fixes use
source-sized fixture tiles, retained candidate vectors and explicit BlockSpecs.
The registered arena was not enlarged. These failures occurred before numerical
invocation and are not omitted from the evidence record.

[`stable-order-provenance.json`](../measurements/lowering-2026-09-13/stable-order-provenance.json)
records measured source `90e6b92`, installed source `2cbf38b`, library `f53538a`,
canonical replication `e974482`, caller `ead87c6` and the exact commands. CPU and
Metal each configured 8046 functions and completed 16027 submissions. The paired
Metal workflow configured 7978 rank-zero functions and completed 15334 rank-zero
submissions. Every terminal runtime code and process exit code is zero; the peer
exited zero after SIGTERM. Both bridges returned to ready, unpaired, zero-client
state with the unchanged arena geometry.

Each successful run contains 36 ordering observations and four alias observations.
Stable ordinals and gathered values match their references for every covered
case and both reuse generations. Source-dependent numerical work completes
before the late axis extent arrives; the complete ready row and its consumers
finish with the other row unpublished. In the paired alias case, one source
extent crosses the link, drives arithmetic on the M4 Pro, and returns its two
output columns before the other extent is published. The second generation
reverses the early source extent and also completes correctly.

Alias early-observation intervals have count 2 each: CPU mean 0.1785 ms and sample
variance 0.000067094528 ms²; Metal mean 0.162125 ms and variance 0.000931651778 ms²;
paired mean 0.718333 ms and variance 0.014379366528 ms². Per-dtype/length ordering
statistics, raw logs, full traces and setup failures are archived. These host
intervals establish the covered progress behavior, not physical overlap duration
or a matched speedup. Ordering side cases run on rank zero; the new fragmented
alias arithmetic runs on rank one in the paired workflow.

The last Xonotic private numerical emitter is now removed. General indexed
native contraction coverage, optimal integer kernels, fusion/storage/placement,
empty-domain gaps and the nine-step plan's matched performance acceptance remain
open. This increment does not redefine those requirements as completed.


## Composable indexed contractions — September 14

`7990090` retains indexed product sums before logical loads are flattened, so
native contraction recognition survives untransposed statistics and pointwise
composition. The shared region cache retains source identities, origins and
arithmetic context. An explicitly returned contraction and its epilogue consume
the same result pages. Existing neighborhood Q/K and G/V statistics benefit
without a caller-specific emitter or new runtime interface.

`ad4dd02` corrects a launch regression found in the new static case. Backing cuts
split its K=5 contraction into more native panels than the existing compiled
reduction required. Setup now compares complete native launch counts with the
retained compiled region plan and retains compiled lowering when the new static
plan adds launches. The fallback writes the requested result directly instead
of adding an identity copy. This conservative policy is not the measured
operation/shape/backend selector still required by the nine-step plan.

The matched Metal fixture uses example `14ad407` in both runs: the baseline
installed library was `f53538a`; the corrected library is `ad4dd02`, installed
from `d6528f6`. Exact function intervals retained by the existing workflow show:

| Indexed case | Baseline functions / submissions | Corrected functions / submissions |
| --- | ---: | ---: |
| Selected F32 weights | 124 / 262 | 94 / 190 |
| Static F32 weights | 24 / 48 | 24 / 48 |
| Selected F16 weights | 124 / 262 | 94 / 190 |

The first native static implementation used 48 functions and 96 submissions;
its log and trace are retained alongside the correction. Two earlier fixture
errors are also retained: a fixed-width index vector did not match a ragged
output extent, and the expected integer-context result incorrectly treated a
nested floating sum like a bare sum into an integer destination. The corrected
fixture observes the existing outputs `(0, 1, 1)` for bare integer reduction,
nested floating reduction and explicit post-reduction conversion.

Each new case uses canonical source, weight and bias pages, ragged K and feature
panels, and two reuse generations. A contraction result completes while its bias
is absent; after that row's bias is published, its epilogue completes while the
other source row remains unpublished. The next generation reverses the early
row. Existing neighborhood, selected contraction and gold cases retain their
own partial-input and distributed progress observations.

[`composable-indexed-provenance.json`](../measurements/lowering-2026-09-13/composable-indexed-provenance.json)
records exact commands, installed and source revisions, raw logs/traces, failed
fixtures, function intervals and online count/mean/sample variance. Corrected
CPU and Metal runs each configured 8214 functions and completed 16296
submissions. Paired Metal configured 8146 rank-zero functions and completed
15603 submissions; the peer configured 1189 and completed 762. All runtime
codes and process exits were zero. The peer exited zero after SIGTERM, and
both bridges returned to ready, unpaired, zero-client state with the unchanged
registered arena.

For the corrected local Metal run, early contraction observation intervals have
count 2 each: selected F32 mean 1.0932915 ms, sample variance 0.0148493314445 ms²;
static F32 mean 0.360625 ms, variance 0.000000587528 ms²; selected F16 mean
1.119375 ms, variance 0.021355417778 ms². These small samples do not establish a
speedup. New contraction cases run on rank zero; existing paired gold, fanout
and fragmented-alias operations exercise the actual RDMA substrate and M4 Pro.
No separate rank-one timing claim is made for the new contraction cases.

The next concrete caller gap is expert weight gradients:
`dW[e,d,h] = sum_{n: selected[n]=e} X[n,d] * G[n,h]` currently reaches the
bounded scalar indexed-add segment owner. Its group ordinal intervals and
outer-product indexing maps should remain explicit for native contiguous-run
or indirect grouped-contraction lowering. Arbitrary group ordinals cannot be
passed to ordinary BLAS/MPS by pretending their backing is contiguous. The
existing [active segment domains](algorithm-sources.md#active-segment-domains)
and [composable contraction sources](algorithm-sources.md#composable-indexed-contractions)
provide the relevant representation and grouped-multiplication prior art.
Measured backend shape selection, broader geometry, integer optimization,
empty-domain gaps, fusion/storage/placement and the full matched performance
acceptance remain open; the nine-step goal remains active.


## Parallel grouped segment reductions — September 14

`5138624` extends the existing segment emitter's Metal lane mapping. Narrow
outputs previously assigned one lane per feature and accumulated all group
ordinals serially in that lane. Setup now divides the SIMD group into feature
and contribution coordinates; register-local partials combine through a fixed
shuffle-XOR tree. The static contribution capacity limits unused reduction lanes.
FP32 shuffles reinterpret bits, and U64 shuffles reconstruct both U32 halves
before modular addition. All lanes participate in each shuffle; ragged feature
lanes omit numerical loads and stores. CPU, single-contribution and wide-panel
source paths retain their prior mapping.

This changes no operand binding, selector, active-domain disposition, launch,
partial allocation or publication boundary. It uses the retained group ordinal
vector and source layouts directly. It does not yet provide matrix-engine
outer-product reuse or a measured backend selector. Mechanism and literature
are in [grouped segment reductions](algorithm-sources.md#grouped-segment-reductions).

`58dcdd3` extends the existing workflow with a 67-row grouped outer product,
33-row input chunks, five experts and 5-by-7 outputs. The earlier expert fixture
had only one row per chunk and could not exercise parallel reduction within a
group. The new case includes repeated interleaved destinations inside a chunk,
negative normalization, invalid NaN contributions, a ragged final chunk and two
reuse generations. Four expert outputs complete while row 66 is absent; its
corresponding expert completes once the final row arrives. All outputs are
compared with valid-domain float64 sums.

Matched Metal baseline and replacement use example `58dcdd3`, with installed
libraries `ad4dd02` and `5138624` respectively. Both configure 8424 functions and
complete 16716 submissions; the grouped case itself retains exactly 210 functions
and 420 submissions. Host intervals are mixed at this small shape and do not
establish a speedup. The source establishes removal of serial contribution work
from narrow panels, not universal performance superiority.

[`grouped-segment-provenance.json`](../measurements/lowering-2026-09-13/grouped-segment-provenance.json)
records exact commands and raw evidence. CPU also configured 8424 functions and
completed 16716 submissions. Paired Metal configured 8356 rank-zero functions and
completed 16023 submissions; the peer configured 1189 and completed 762. Every
runtime code and process exit was zero. After SIGTERM and the peer's zero exit,
both bridges were ready, unpaired and without clients. Arena geometry remained
unchanged.

Early unrelated-destination observation intervals each have count 2: baseline
Metal mean 4.809479 ms, sample variance 0.138579012882 ms²; replacement Metal
mean 4.829854 ms, variance 0.314391194882 ms²; CPU mean 2.3054585 ms, variance
0.2513993504445 ms²; paired mean 4.7835415 ms, variance 0.0939615585005 ms².
These are host observations, not physical overlap duration. The grouped side
case executes on rank zero; no separate M4 Pro grouped-kernel claim is made.

The next measured-selection requirement has a concrete existing owner:
`MeshFunction` retains setup bindings, `submit_ready` records dispatch timestamps,
`complete_part` observes each numerical completion, and `Program.trace` exposes
the record. Today the record keeps only latest timestamps and cumulative
submissions, which cannot recover per-call variance. Extend those owners with
immutable operation/shape/dtype/layout/backend descriptors and online timing
moments; retain CPU SGEMM/NEON/compiled and Metal MPS/compiled identities. Keep
host delay/execution and GPU service statistics separate and account explicitly
for errors and omitted functions. Archived profiles can then inform setup
selection of complete plans including merges/assembly. This does not require
another evaluator, invocation-time configuration or participant scheduler.

Matrix-engine grouped operand reuse and measured selection remain incomplete,
alongside broader geometry, integer optimization, empty-domain gaps,
fusion/storage/placement and the full performance acceptance. This increment
leaves the nine-step goal active.
